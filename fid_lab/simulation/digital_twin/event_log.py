"""Observable append-only event authority with idempotency and watermark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock
import torch
import zstandard

from fid_lab.launches.release_resources import file_sha256

from .contracts import AppEventBatch


EVENT_LOG_SCHEMA = "observable-app-events-v6"


@dataclass(frozen=True)
class EventPartitionRef:
    sequence: int
    object_sha256: str
    rows: int
    minimum_event_time: int
    maximum_event_time: int
    minimum_ingest_time: int
    maximum_ingest_time: int


class ObservableEventLog:
    def __init__(
        self,
        allowed_lateness: int = 0,
        *,
        root: Path | None = None,
    ):
        if allowed_lateness < 0:
            raise ValueError("allowed lateness cannot be negative")
        self._batches: list[AppEventBatch] = []
        self._ids_by_event_time: dict[int, torch.Tensor] = {}
        self._events = 0
        self._allowed_lateness = allowed_lateness
        self._ingest_watermark = -1
        self.root = root
        self._partition_refs: list[EventPartitionRef] = []
        if root is not None:
            self.objects = root / "objects"
            self.manifest_path = root / "event-log.json"
            self.lock_path = root / "event-log.lock"
            self.objects.mkdir(parents=True, exist_ok=True)

    @property
    def allowed_lateness(self) -> int:
        return self._allowed_lateness

    @property
    def watermark(self) -> int:
        if self._ingest_watermark < 0:
            return -1
        return max(-1, self._ingest_watermark - self._allowed_lateness)

    @property
    def ingest_watermark(self) -> int:
        return self._ingest_watermark

    def append(self, batch: AppEventBatch) -> None:
        staged = self.validate(batch)
        self._ids_by_event_time.update(staged)
        self._events += len(batch.event_id)
        retained = self._to_cpu(batch) if self.root is not None else batch
        if self.root is None:
            self._batches.append(retained)
        elif len(retained.event_id):
            self._append_partition(retained)
            self._batches.append(retained)
        if len(batch.ingest_time):
            self._ingest_watermark = max(
                self._ingest_watermark, int(batch.ingest_time.max())
            )
        self._evict_cold_state()
        if self.root is not None:
            self._write_manifest()

    def validate(self, batch: AppEventBatch) -> dict[int, torch.Tensor]:
        if len(batch.ingest_time) and (
            batch.ingest_time < self._ingest_watermark
        ).any():
            raise ValueError("event log delivery time cannot move backwards")
        if len(batch.event_time) and self.watermark >= 0 and (
            batch.event_time < self.watermark
        ).any():
            raise ValueError("event log delivery is older than allowed lateness")
        staged: dict[int, torch.Tensor] = {}
        for event_time in torch.unique(batch.event_time).tolist():
            selected = batch.event_time == event_time
            incoming = batch.event_id[selected].detach().cpu()
            existing = self._ids_by_event_time.get(event_time)
            if existing is None:
                staged[event_time] = torch.sort(incoming).values
                continue
            merged = torch.cat((existing, incoming))
            unique = torch.unique(merged, sorted=True)
            duplicate_count = len(merged) - len(unique)
            if duplicate_count:
                raise ValueError(
                    f"event log duplicate ids: {duplicate_count}"
                )
            staged[event_time] = unique
        return staged

    def read(
        self,
        *,
        through: int | None = None,
        ingested_through: int | None = None,
    ) -> AppEventBatch:
        batches = (
            self._read_partitions(through, ingested_through)
            if self.root is not None else tuple(self._batches)
        )
        device = (
            batches[0].event_id.device
            if batches else torch.device("cpu")
        )
        result = AppEventBatch.concatenate(batches) if batches else (
            AppEventBatch.empty(device)
        )
        selected = torch.ones_like(result.event_id, dtype=torch.bool)
        if through is not None:
            selected &= result.event_time <= through
        if ingested_through is not None:
            selected &= result.ingest_time <= ingested_through
        return result.select(selected)

    def partitions(self) -> tuple[AppEventBatch, ...]:
        """Return immutable batch references in authoritative append order."""
        if self.root is None:
            return tuple(self._batches)
        return self._read_partitions(None, None)

    @property
    def durable(self) -> bool:
        return self.root is not None

    def checkpoint_partitions(self) -> tuple[dict[str, object], ...]:
        return tuple(asdict(ref) for ref in self._partition_refs)

    def restore_partitions(
        self,
        refs: tuple[dict[str, object], ...],
        expected_manifest: dict[str, object],
    ) -> None:
        if self.root is None:
            raise ValueError("durable event restore requires a partition root")
        parsed = [EventPartitionRef(**value) for value in refs]
        for ref in parsed:
            self._verify_object(ref.object_sha256)
        self._partition_refs = parsed
        self._events = int(expected_manifest["events"])
        self._ingest_watermark = int(expected_manifest["ingest_watermark"])
        self._batches.clear()
        self._ids_by_event_time.clear()
        hot_start = self.watermark
        for batch in self._read_partitions(None, None):
            if len(batch.ingest_time) and int(batch.ingest_time.max()) >= hot_start:
                self._batches.append(batch)
                for event_time in torch.unique(batch.event_time).tolist():
                    selected = batch.event_time == event_time
                    incoming = batch.event_id[selected]
                    existing = self._ids_by_event_time.get(event_time)
                    self._ids_by_event_time[event_time] = torch.unique(
                        incoming if existing is None else torch.cat((existing, incoming)),
                        sorted=True,
                    )
        if self.manifest() != expected_manifest:
            raise ValueError("restored durable event log differs from checkpoint")
        self._write_manifest()

    def manifest(self) -> dict[str, int | str]:
        value = {
            "schema": EVENT_LOG_SCHEMA,
            "events": self._events,
            "batches": (
                len(self._partition_refs) if self.durable else len(self._batches)
            ),
            "watermark": self.watermark,
            "ingest_watermark": self._ingest_watermark,
            "allowed_lateness": self._allowed_lateness,
            "durable": int(self.durable),
            "hot_batches": len(self._batches),
        }
        if self.root is not None:
            value["partition_stream_sha256"] = self._partition_stream_hash()
        return value

    def _append_partition(self, batch: AppEventBatch) -> None:
        with NamedTemporaryFile(dir=self.objects, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with zstandard.ZstdCompressor(level=3).stream_writer(
                    stream, closefd=False,
                ) as compressed:
                    torch.save(batch, compressed)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        digest = file_sha256(temporary)
        target = self.objects / f"{digest}.pt.zst"
        if target.exists():
            self._verify_object(digest)
            temporary.unlink()
        else:
            os.replace(temporary, target)
        self._partition_refs.append(EventPartitionRef(
            sequence=len(self._partition_refs),
            object_sha256=digest,
            rows=len(batch.event_id),
            minimum_event_time=int(batch.event_time.min()) if len(batch.event_time) else -1,
            maximum_event_time=int(batch.event_time.max()) if len(batch.event_time) else -1,
            minimum_ingest_time=int(batch.ingest_time.min()) if len(batch.ingest_time) else -1,
            maximum_ingest_time=int(batch.ingest_time.max()) if len(batch.ingest_time) else -1,
        ))

    def _read_partitions(
        self,
        through: int | None,
        ingested_through: int | None,
    ) -> tuple[AppEventBatch, ...]:
        result = []
        for ref in self._partition_refs:
            if through is not None and ref.minimum_event_time > through:
                continue
            if (
                ingested_through is not None
                and ref.minimum_ingest_time > ingested_through
            ):
                continue
            result.append(self._read_object(ref.object_sha256))
        return tuple(result)

    def _read_object(self, digest: str) -> AppEventBatch:
        path = self.objects / f"{digest}.pt.zst"
        self._verify_object(digest)
        with path.open("rb") as source, NamedTemporaryFile() as raw:
            zstandard.ZstdDecompressor().copy_stream(source, raw)
            raw.flush()
            raw.seek(0)
            value = torch.load(raw, map_location="cpu", weights_only=False)
        if not isinstance(value, AppEventBatch):
            raise ValueError("event partition object has an invalid type")
        return value

    def _verify_object(self, digest: str) -> None:
        path = self.objects / f"{digest}.pt.zst"
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("event partition object is missing or corrupted")

    def _evict_cold_state(self) -> None:
        if self.root is None or self.watermark < 0:
            return
        self._batches = [
            batch for batch in self._batches
            if not len(batch.ingest_time)
            or int(batch.ingest_time.max()) >= self.watermark
        ]
        self._ids_by_event_time = {
            event_time: ids
            for event_time, ids in self._ids_by_event_time.items()
            if event_time >= self.watermark
        }

    def _partition_stream_hash(self) -> str:
        payload = json.dumps(
            [asdict(ref) for ref in self._partition_refs],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return sha256(payload).hexdigest()

    def _write_manifest(self) -> None:
        value = {
            "manifest": self.manifest(),
            "partitions": [asdict(ref) for ref in self._partition_refs],
        }
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
        with FileLock(str(self.lock_path)):
            with NamedTemporaryFile(
                mode="w", dir=self.root, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            os.replace(temporary, self.manifest_path)

    @staticmethod
    def _to_cpu(batch: AppEventBatch) -> AppEventBatch:
        return AppEventBatch(**{
            field.name: getattr(batch, field.name).detach().cpu()
            for field in fields(batch)
        })
