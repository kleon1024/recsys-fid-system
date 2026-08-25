"""Immutable request-level facts emitted by the evolving factual world.

World checkpoints restore mutable ecosystem state. This stream separately owns
the request, candidate, point-in-time context and response facts needed to
rebuild training examples at a later label watermark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile
from typing import Mapping

from filelock import FileLock
import torch

from ..checkpoint import WorldBranchRef
from ..contracts import AppEventBatch
from ..engine import LayerAssignmentTrace, TickResult
from ..platform.projection import ProjectionSnapshot
from ..samples.contracts import RequestCandidateTrace, RequestContextBatch


FACTUAL_REQUEST_STREAM_SCHEMA = "factual-request-stream/v2"
READABLE_REQUEST_STREAM_SCHEMAS = frozenset({
    "factual-request-stream/v1",
    FACTUAL_REQUEST_STREAM_SCHEMA,
})


@dataclass(frozen=True)
class FactualRequestPartitionRef:
    logical_time: int
    object_sha256: str
    requests: int
    events: int
    trace_manifest_sha256: str
    world_manifest_sha256: str = ""


@dataclass(frozen=True)
class FactualRequestPartition:
    logical_time: int
    trace: RequestCandidateTrace
    context: RequestContextBatch
    events: AppEventBatch
    projection: ProjectionSnapshot
    layer_assignment: LayerAssignmentTrace | None
    world_manifest: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logical_time < 0:
            raise ValueError("request partition logical time cannot be negative")
        if not torch.equal(self.trace.request_id, self.context.request_id):
            raise ValueError("request partition trace and context differ")
        if len(self.trace.event_time) and not (
            self.trace.event_time == self.logical_time
        ).all():
            raise ValueError("request partition contains another event time")
        if self.projection.as_of_ingest_time < self.logical_time:
            raise ValueError("request partition projection predates requests")
        if self.layer_assignment is not None and not torch.equal(
            self.trace.request_id, self.layer_assignment.request_id,
        ):
            raise ValueError("request partition layer assignment differs")


class FactualRequestStream:
    """Content-verified append-only stream scoped to one world branch."""

    def __init__(self, root: Path, branch: WorldBranchRef):
        self.root = root
        self.branch = branch
        self.objects = root / "objects"
        self.staging = root / "staging"
        self.manifest_path = root / "request-stream.json"
        self.lock_path = root / "request-stream.lock"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)

    def _partition_payload(
        self,
        tick: TickResult,
        projection: ProjectionSnapshot,
        world_manifest: Mapping[str, str],
    ) -> tuple[FactualRequestPartitionRef, bytes]:
        if tick.candidate_trace is None or tick.request_context is None:
            raise ValueError("factual request stream requires a complete trace")
        partition = FactualRequestPartition(
            logical_time=tick.logical_time,
            trace=tick.candidate_trace,
            context=tick.request_context,
            events=AppEventBatch.concatenate((
                tick.entry_events,
                tick.response_events,
            )),
            projection=projection,
            layer_assignment=tick.layer_assignment,
            world_manifest=dict(world_manifest),
        )
        payload = self._serialize(partition)
        digest = sha256(payload).hexdigest()
        return FactualRequestPartitionRef(
            logical_time=tick.logical_time,
            object_sha256=digest,
            requests=len(partition.trace.request_id),
            events=len(partition.events.event_id),
            trace_manifest_sha256=self._trace_manifest_hash(partition.trace),
            world_manifest_sha256=self._world_manifest_hash(
                partition.world_manifest,
            ),
        ), payload

    def append(
        self,
        tick: TickResult,
        projection: ProjectionSnapshot,
        world_manifest: Mapping[str, str],
    ) -> FactualRequestPartitionRef:
        ref, payload = self._partition_payload(
            tick, projection, world_manifest,
        )
        digest = ref.object_sha256
        with FileLock(str(self.lock_path)):
            manifest = self._read_manifest()
            if manifest["schema"] != FACTUAL_REQUEST_STREAM_SCHEMA:
                raise ValueError("legacy request streams are immutable")
            existing = manifest["partitions"].get(str(tick.logical_time))
            if existing is not None:
                if existing != asdict(ref):
                    raise ValueError("request partition event time changed content")
                self._verify_object(digest)
                return ref
            self._write_object(digest, payload)
            manifest["partitions"][str(tick.logical_time)] = asdict(ref)
            manifest["partitions"] = dict(sorted(
                manifest["partitions"].items(), key=lambda row: int(row[0]),
            ))
            manifest["stream_sha256"] = self._stream_hash(manifest)
            self._write_json(manifest)
        return ref

    def stage(
        self,
        transaction_id: str,
        tick: TickResult,
        projection: ProjectionSnapshot,
        world_manifest: Mapping[str, str],
    ) -> FactualRequestPartitionRef:
        """Write an unpublished partition owned by one launch attempt."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", transaction_id):
            raise ValueError("request transaction id is unsafe")
        ref, payload = self._partition_payload(
            tick, projection, world_manifest,
        )
        directory = self.staging / transaction_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{ref.logical_time}-{ref.object_sha256}.pt"
        if target.exists():
            if sha256(target.read_bytes()).hexdigest() != ref.object_sha256:
                raise ValueError("staged request partition is corrupted")
            return ref
        with NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, target)
        return ref

    def commit_staged(
        self,
        transaction_id: str,
        refs: tuple[FactualRequestPartitionRef, ...],
    ) -> None:
        """Publish all staged partitions with one manifest replacement."""
        if len({ref.logical_time for ref in refs}) != len(refs):
            raise ValueError("request transaction repeats a logical time")
        directory = self.staging / transaction_id
        with FileLock(str(self.lock_path)):
            manifest = self._read_manifest()
            for ref in refs:
                staged = directory / (
                    f"{ref.logical_time}-{ref.object_sha256}.pt"
                )
                if (
                    not staged.is_file()
                    or sha256(staged.read_bytes()).hexdigest()
                    != ref.object_sha256
                ):
                    raise ValueError("staged request partition is missing")
                existing = manifest["partitions"].get(str(ref.logical_time))
                if existing is not None and existing != asdict(ref):
                    raise ValueError(
                        "request partition event time changed content"
                    )
            for ref in refs:
                staged = directory / (
                    f"{ref.logical_time}-{ref.object_sha256}.pt"
                )
                target = self.objects / f"{ref.object_sha256}.pt"
                if target.exists():
                    self._verify_object(ref.object_sha256)
                    staged.unlink()
                else:
                    os.replace(staged, target)
                manifest["partitions"][str(ref.logical_time)] = asdict(ref)
            manifest["partitions"] = dict(sorted(
                manifest["partitions"].items(), key=lambda row: int(row[0]),
            ))
            manifest["stream_sha256"] = self._stream_hash(manifest)
            self._write_json(manifest)
        shutil.rmtree(directory, ignore_errors=True)

    def abort_staged(self, transaction_id: str) -> None:
        shutil.rmtree(self.staging / transaction_id, ignore_errors=True)

    def reconcile_through(
        self, checkpoint_logical_time: int,
    ) -> tuple[FactualRequestPartitionRef, ...]:
        """Unpublish partitions not backed by the factual branch head."""
        with FileLock(str(self.lock_path)):
            manifest = self._read_manifest()
            orphaned = tuple(
                FactualRequestPartitionRef(**value)
                for key, value in manifest["partitions"].items()
                if int(key) > checkpoint_logical_time
            )
            if not orphaned:
                return ()
            manifest["partitions"] = {
                key: value
                for key, value in manifest["partitions"].items()
                if int(key) <= checkpoint_logical_time
            }
            manifest["stream_sha256"] = self._stream_hash(manifest)
            self._write_json(manifest)
        return orphaned

    def refs(self, *, training: bool = False) -> tuple[FactualRequestPartitionRef, ...]:
        if training and not self.branch.training_authority:
            raise ValueError("diagnostic world requests cannot train a model")
        with FileLock(str(self.lock_path)):
            manifest = self._read_manifest()
            refs = tuple(
                FactualRequestPartitionRef(**value)
                for value in manifest["partitions"].values()
            )
            for ref in refs:
                self._verify_object(ref.object_sha256)
        return refs

    def read(
        self,
        ref: FactualRequestPartitionRef,
        *,
        device: str | torch.device | None = None,
    ) -> FactualRequestPartition:
        self._verify_object(ref.object_sha256)
        value = torch.load(
            self.objects / f"{ref.object_sha256}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(value, FactualRequestPartition):
            raise ValueError("request partition object has an invalid type")
        actual = FactualRequestPartitionRef(
            logical_time=value.logical_time,
            object_sha256=ref.object_sha256,
            requests=len(value.trace.request_id),
            events=len(value.events.event_id),
            trace_manifest_sha256=self._trace_manifest_hash(value.trace),
            world_manifest_sha256=self._world_manifest_hash(
                getattr(value, "world_manifest", {}),
            ),
        )
        if actual != ref:
            raise ValueError("request partition metadata differs from object")
        if device is None:
            return value
        moved = _to_device(value, torch.device(device))
        if not isinstance(moved, FactualRequestPartition):
            raise TypeError("request partition device transfer changed type")
        return moved

    @property
    def stream_sha256(self) -> str:
        with FileLock(str(self.lock_path)):
            return str(self._read_manifest()["stream_sha256"])

    def _read_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            manifest = {
                "schema": FACTUAL_REQUEST_STREAM_SCHEMA,
                "branch": self.branch.name,
                "training_authority": self.branch.training_authority,
                "partitions": {},
            }
            manifest["stream_sha256"] = self._stream_hash(manifest)
            return manifest
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("schema") not in READABLE_REQUEST_STREAM_SCHEMAS:
            raise ValueError("request stream schema is unsupported")
        if manifest.get("branch") != self.branch.name:
            raise ValueError("request stream belongs to another branch")
        if manifest.get("training_authority") != self.branch.training_authority:
            raise ValueError("request stream training authority changed")
        if manifest.get("stream_sha256") != self._stream_hash(manifest):
            raise ValueError("request stream manifest hash differs")
        return manifest

    def _write_object(self, digest: str, payload: bytes) -> None:
        path = self.objects / f"{digest}.pt"
        if path.exists():
            self._verify_object(digest)
            return
        with NamedTemporaryFile(dir=self.objects, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)

    def _verify_object(self, digest: str) -> None:
        path = self.objects / f"{digest}.pt"
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("request partition object is missing or corrupted")

    def _write_json(self, value: dict[str, object]) -> None:
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
        with NamedTemporaryFile(
            mode="w", dir=self.root, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, self.manifest_path)

    @staticmethod
    def _serialize(value: FactualRequestPartition) -> bytes:
        stream = BytesIO()
        torch.save(value, stream)
        return stream.getvalue()

    @staticmethod
    def _trace_manifest_hash(trace: RequestCandidateTrace) -> str:
        payload = json.dumps(
            asdict(trace.manifest), sort_keys=True, separators=(",", ":"),
        ).encode()
        return sha256(payload).hexdigest()

    @staticmethod
    def _world_manifest_hash(manifest: Mapping[str, str]) -> str:
        if not manifest:
            return ""
        payload = json.dumps(
            dict(manifest), sort_keys=True, separators=(",", ":"),
        ).encode()
        return sha256(payload).hexdigest()

    @staticmethod
    def _stream_hash(manifest: dict[str, object]) -> str:
        identity = {
            key: value for key, value in manifest.items()
            if key != "stream_sha256"
        }
        payload = json.dumps(
            identity, sort_keys=True, separators=(",", ":"),
        ).encode()
        return sha256(payload).hexdigest()


def _to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{
            field.name: _to_device(getattr(value, field.name), device)
            for field in fields(value)
        })
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _to_device(item, device) for key, item in value.items()
        }
    return value
