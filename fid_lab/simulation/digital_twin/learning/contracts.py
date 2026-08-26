"""Content identities shared by sample lanes, trainers and registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path

import torch


def content_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def learning_source_hash() -> str:
    """Bind artifacts to the current learning and online feature source closure."""
    digital_twin = Path(__file__).resolve().parents[1]
    files = tuple(sorted((digital_twin / "learning").rglob("*.py"))) + tuple(
        sorted((digital_twin / "platform" / "features").glob("*.py"))
    ) + (
        digital_twin / "platform" / "ranking.py",
        digital_twin / "platform" / "retrieval.py",
    )
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(digital_twin)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class Lane(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class LaneCursor:
    lane: Lane
    contract_hash: str
    consumed: tuple[tuple[str, str], ...] = ()
    event_watermark: int = -1

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "v4-learning-lane-cursor-v1",
            "lane": self.lane.value,
            "contract_hash": self.contract_hash,
            "consumed": [
                {"partition_key": key, "content_sha256": digest}
                for key, digest in self.consumed
            ],
            "event_watermark": self.event_watermark,
        }

    @classmethod
    def from_manifest(cls, value: dict[str, object]) -> LaneCursor:
        if value.get("schema") != "v4-learning-lane-cursor-v1":
            raise ValueError("lane cursor schema is unsupported")
        return cls(
            lane=Lane(str(value["lane"])),
            contract_hash=str(value["contract_hash"]),
            consumed=tuple(
                (str(row["partition_key"]), str(row["content_sha256"]))
                for row in value["consumed"]
            ),
            event_watermark=int(value["event_watermark"]),
        )


@dataclass(frozen=True)
class ArtifactCompatibility:
    dataset_contract_hash: str
    feature_manifest_hash: str
    stage_contract_hash: str
    feature_version: str
    fid_version: str
    catalog_version: str
    index_version: str
    corpus_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not value:
                raise ValueError(f"artifact compatibility requires {name}")

    def manifest(self) -> dict[str, str]:
        return asdict(self)

    @property
    def compatibility_hash(self) -> str:
        return content_hash(self.manifest())


@dataclass(frozen=True)
class ServingCompatibility:
    """Fields that the online scorer can independently verify."""

    feature_manifest_hash: str
    feature_version: str
    fid_version: str
    catalog_version: str
    index_version: str
    code_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not value:
                raise ValueError(f"serving compatibility requires {name}")

    def validate(self, artifact: ArtifactCompatibility) -> None:
        for name, value in asdict(self).items():
            if getattr(artifact, name) != value:
                raise ValueError(f"serving {name} differs from model artifact")


@dataclass(frozen=True)
class ProbeBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    request_time: torch.Tensor
    item_id: torch.Tensor
    dwell_ms: torch.Tensor
    dense_features: torch.Tensor
    sparse_buckets: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    joint_logging_probability: torch.Tensor
    task_names: tuple[str, ...]
    dense_feature_names: tuple[str, ...]
    sparse_feature_names: tuple[str, ...]
    feature_manifest_hash: str
    partition_content_hashes: tuple[str, ...]
    event_watermark: int

    def __post_init__(self) -> None:
        rows = len(self.request_id)
        for name in (
            "user_id", "surface", "request_time", "item_id",
            "joint_logging_probability", "dwell_ms",
        ):
            if getattr(self, name).shape != (rows,):
                raise ValueError(f"probe {name} is not row aligned")
        for name in ("dense_features", "sparse_buckets"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != rows:
                raise ValueError(f"probe {name} is not row aligned")
        if self.labels.shape != self.label_mask.shape:
            raise ValueError("probe labels and masks differ")
        if self.labels.shape != (rows, len(self.task_names)):
            raise ValueError("probe task contract differs from labels")
        if self.dense_features.shape[1] != len(self.dense_feature_names):
            raise ValueError("probe dense feature names differ")
        if self.sparse_buckets.shape[1] != len(self.sparse_feature_names):
            raise ValueError("probe sparse feature names differ")
        if not self.feature_manifest_hash or not self.partition_content_hashes:
            raise ValueError("probe batch lineage is incomplete")
