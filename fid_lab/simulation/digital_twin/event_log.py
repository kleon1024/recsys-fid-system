"""Observable append-only event authority with idempotency and watermark."""

from __future__ import annotations

import torch

from .contracts import AppEventBatch


class ObservableEventLog:
    def __init__(self, allowed_lateness: int = 0):
        if allowed_lateness < 0:
            raise ValueError("allowed lateness cannot be negative")
        self._batches: list[AppEventBatch] = []
        self._ids_by_event_time: dict[int, torch.Tensor] = {}
        self._events = 0
        self._allowed_lateness = allowed_lateness
        self._ingest_watermark = -1

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
        self._batches.append(batch)
        if len(batch.ingest_time):
            self._ingest_watermark = max(
                self._ingest_watermark, int(batch.ingest_time.max())
            )

    def validate(self, batch: AppEventBatch) -> dict[int, torch.Tensor]:
        if len(batch.ingest_time) and (
            batch.ingest_time < self._ingest_watermark
        ).any():
            raise ValueError("event log delivery time cannot move backwards")
        staged: dict[int, torch.Tensor] = {}
        for event_time in torch.unique(batch.event_time).tolist():
            selected = batch.event_time == event_time
            incoming = batch.event_id[selected]
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
        result = AppEventBatch.concatenate(self._batches)
        selected = torch.ones_like(result.event_id, dtype=torch.bool)
        if through is not None:
            selected &= result.event_time <= through
        if ingested_through is not None:
            selected &= result.ingest_time <= ingested_through
        return result.select(selected)

    def partitions(self) -> tuple[AppEventBatch, ...]:
        """Return immutable batch references in authoritative append order."""
        return tuple(self._batches)

    def manifest(self) -> dict[str, int | str]:
        return {
            "schema": "observable-app-events-v5",
            "events": self._events,
            "batches": len(self._batches),
            "watermark": self.watermark,
            "ingest_watermark": self._ingest_watermark,
            "allowed_lateness": self._allowed_lateness,
        }
