"""Persistent content-bound candidate, active and fallback model registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from filelock import FileLock
import torch

from ..observability import CheckpointRecord, replace_json_atomic
from .contracts import ArtifactCompatibility, Lane, ServingCompatibility
from .probe import ProbeArtifact


REGISTRY_SCHEMA = "v4-persistent-model-registry-v2"


class RegistryArtifact(Protocol):
    feature_manifest_hash: str
    model_name: str

    def checkpoint(self) -> dict[str, object]: ...

    def validate_compatibility(self, expected: ArtifactCompatibility) -> None: ...


@dataclass(frozen=True)
class ModelRecord:
    model_name: str
    serving_version_id: int
    checkpoint_version: str
    artifact_file: str
    artifact_sha256: str
    training_lane: str
    status: str
    validation_status: str
    data_watermark: int
    parent_version_id: int
    fallback_version_id: int
    compatibility: dict[str, str]

    @property
    def compatibility_hash(self) -> str:
        return ArtifactCompatibility(**self.compatibility).compatibility_hash


class PersistentModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = root / "artifacts"
        self.path = root / "registry.json"
        self.lock = FileLock(str(root / ".registry.lock"))
        self.artifacts.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            replace_json_atomic(self.path, {
                "schema": REGISTRY_SCHEMA,
                "next_version": 1,
                "aliases": {},
                "models": {},
            })
        self._state()

    def _state(self) -> dict[str, object]:
        value = json.loads(self.path.read_text())
        if value.get("schema") != REGISTRY_SCHEMA:
            raise ValueError("model registry schema is unsupported")
        return value

    @staticmethod
    def _record(value: dict[str, object]) -> ModelRecord:
        return ModelRecord(**value)

    def record(self, version: int) -> ModelRecord:
        value = self._state()["models"].get(str(version))
        if value is None:
            raise KeyError(f"unknown model version: {version}")
        return self._record(value)

    def alias(self, name: str) -> ModelRecord | None:
        version = self._state()["aliases"].get(name)
        return None if version is None else self.record(int(version))

    def register_candidate(
        self,
        artifact: RegistryArtifact,
        compatibility: ArtifactCompatibility,
        *,
        lane: Lane,
        data_watermark: int,
    ) -> ModelRecord:
        artifact.validate_compatibility(compatibility)
        with self.lock:
            state = self._state()
            version = int(state["next_version"])
            temporary = self._save_temporary(artifact)
            digest = sha256(temporary.read_bytes()).hexdigest()
            checkpoint_version = f"{artifact.model_name}-{digest[:20]}"
            target = self.artifacts / f"{checkpoint_version}.pt"
            if target.exists() and sha256(target.read_bytes()).hexdigest() != digest:
                temporary.unlink(missing_ok=True)
                raise ValueError("checkpoint identity collision")
            os.replace(temporary, target)
            parent = state["aliases"].get("active", -1)
            fallback = state["aliases"].get("fallback", -1)
            record = ModelRecord(
                model_name=artifact.model_name,
                serving_version_id=version,
                checkpoint_version=checkpoint_version,
                artifact_file=target.name,
                artifact_sha256=digest,
                training_lane=lane.value,
                status="candidate",
                validation_status="pending",
                data_watermark=data_watermark,
                parent_version_id=int(parent),
                fallback_version_id=int(fallback),
                compatibility=compatibility.manifest(),
            )
            state["models"][str(version)] = asdict(record)
            state["aliases"][f"{lane.value}_latest"] = version
            state["aliases"]["candidate"] = version
            state["next_version"] = version + 1
            replace_json_atomic(self.path, state)
            return record

    def _save_temporary(self, artifact: RegistryArtifact) -> Path:
        with NamedTemporaryFile(
            dir=self.artifacts, prefix=".checkpoint-", suffix=".pt", delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(artifact.checkpoint(), temporary)
        return temporary

    def shadow(self, version: int, *, validation_status: str) -> ModelRecord:
        if validation_status not in {"pass", "hold", "reject"}:
            raise ValueError("unsupported snapshot validation status")
        return self._transition(
            version,
            allowed={"candidate"},
            status="shadow" if validation_status != "reject" else "rejected",
            validation_status=validation_status,
        )

    def review_shadow(
        self, version: int, *, validation_status: str,
    ) -> ModelRecord:
        """Record the factual A/B decision for an existing shadow artifact."""
        if validation_status not in {"pass", "hold", "reject"}:
            raise ValueError("unsupported shadow review status")
        return self._transition(
            version,
            allowed={"shadow"},
            status="shadow" if validation_status != "reject" else "rejected",
            validation_status=validation_status,
        )

    def promote(self, version: int) -> ModelRecord:
        with self.lock:
            state = self._state()
            current = self._record(state["models"][str(version)])
            if current.status != "shadow" or current.validation_status != "pass":
                raise ValueError("only a passed shadow checkpoint can be promoted")
            previous = state["aliases"].get("active")
            if previous is not None:
                prior = self._record(state["models"][str(previous)])
                state["models"][str(previous)] = asdict(
                    self._replace_record(prior, status="retired")
                )
                state["aliases"]["fallback"] = int(previous)
            promoted = self._replace_record(
                current,
                status="active",
                fallback_version_id=int(previous or -1),
            )
            state["models"][str(version)] = asdict(promoted)
            state["aliases"]["active"] = version
            if state["aliases"].get("candidate") == version:
                del state["aliases"]["candidate"]
            replace_json_atomic(self.path, state)
            return promoted

    def reject(self, version: int) -> ModelRecord:
        return self._transition(
            version,
            allowed={"candidate", "shadow"},
            status="rejected",
            validation_status="reject",
        )

    def _transition(
        self,
        version: int,
        *,
        allowed: set[str],
        status: str,
        validation_status: str,
    ) -> ModelRecord:
        with self.lock:
            state = self._state()
            current = self._record(state["models"][str(version)])
            if current.status not in allowed:
                raise ValueError(f"model status cannot transition from {current.status}")
            updated = self._replace_record(
                current, status=status, validation_status=validation_status,
            )
            state["models"][str(version)] = asdict(updated)
            if status == "rejected" and state["aliases"].get("candidate") == version:
                del state["aliases"]["candidate"]
            replace_json_atomic(self.path, state)
            return updated

    @staticmethod
    def _replace_record(record: ModelRecord, **changes: object) -> ModelRecord:
        values = asdict(record)
        values.update(changes)
        return ModelRecord(**values)

    def load(
        self,
        alias: str,
        expected: ArtifactCompatibility,
        *,
        corpus=None,
    ) -> tuple[ModelRecord, RegistryArtifact]:
        record = self.alias(alias)
        if record is None:
            raise ValueError(f"model alias is not assigned: {alias}")
        return self._load_record(record, expected, corpus=corpus)

    def load_version(
        self,
        version: int,
        expected: ArtifactCompatibility,
        *,
        corpus=None,
    ) -> tuple[ModelRecord, RegistryArtifact]:
        return self._load_record(self.record(version), expected, corpus=corpus)

    def load_version_for_serving(
        self,
        version: int,
        expected: ServingCompatibility,
    ) -> tuple[ModelRecord, RegistryArtifact]:
        record = self.record(version)
        artifact_compatibility = ArtifactCompatibility(**record.compatibility)
        expected.validate(artifact_compatibility)
        return self._load_record(record, artifact_compatibility)

    def _load_record(
        self,
        record: ModelRecord,
        expected: ArtifactCompatibility,
        *,
        corpus=None,
    ) -> tuple[ModelRecord, RegistryArtifact]:
        if record.compatibility != expected.manifest():
            raise ValueError("model artifact is incompatible with serving snapshot")
        path = self.artifacts / record.artifact_file
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != (
            record.artifact_sha256
        ):
            raise ValueError("model artifact content hash mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        artifact = self._load_checkpoint(checkpoint, corpus=corpus)
        artifact.validate_compatibility(expected)
        return record, artifact

    @staticmethod
    def _load_checkpoint(checkpoint, *, corpus=None) -> RegistryArtifact:
        schema = checkpoint.get("schema")
        if schema == "v4-lr-infrastructure-probe-v1":
            return ProbeArtifact.from_checkpoint(checkpoint)
        if schema == "v4-retrieval-artifact-v1":
            if corpus is None:
                raise ValueError("retrieval artifact load requires its corpus")
            from .retrieval.artifact import RetrievalArtifact

            return RetrievalArtifact.from_checkpoint(checkpoint, corpus)
        raise ValueError("model checkpoint schema is unsupported")

    def load_active_with_fallback(
        self,
        expected: ArtifactCompatibility,
        *,
        corpus=None,
    ) -> tuple[ModelRecord, RegistryArtifact, bool]:
        try:
            record, artifact = self.load("active", expected, corpus=corpus)
            return record, artifact, False
        except (OSError, ValueError):
            record, artifact = self.load("fallback", expected, corpus=corpus)
            return record, artifact, True

    def checkpoint_records(self, created_time: int) -> tuple[CheckpointRecord, ...]:
        state = self._state()
        records = []
        for value in state["models"].values():
            model = self._record(value)
            compatibility = ArtifactCompatibility(**model.compatibility)
            fallback = (
                str(model.fallback_version_id)
                if model.fallback_version_id >= 0 else ""
            )
            records.append(CheckpointRecord(
                created_time=created_time,
                lane=model.training_lane,
                model_name=model.model_name,
                checkpoint_version=model.checkpoint_version,
                data_watermark=model.data_watermark,
                sample_manifest=compatibility.dataset_contract_hash,
                feature_version=compatibility.feature_version,
                fid_version=compatibility.fid_version,
                index_version=compatibility.index_version,
                validation_status=model.validation_status,
                publish_state=model.status,
                fallback_version=fallback,
                serving_version_id=model.serving_version_id,
                artifact_sha256=model.artifact_sha256,
                compatibility_hash=model.compatibility_hash,
            ))
        return tuple(sorted(records, key=lambda row: row.serving_version_id))
