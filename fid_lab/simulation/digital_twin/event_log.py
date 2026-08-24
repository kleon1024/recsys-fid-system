"""Observable append-only event authority with idempotency and watermark."""

from __future__ import annotations

import torch

from .contracts import AppEventBatch


class ObservableEventLog:
    def __init__(self):
        self._batches: list[AppEventBatch] = []
        self._ids_by_event_time: dict[int, torch.Tensor] = {}
        self._events = 0
        self._watermark = -1

    @property
    def watermark(self) -> int:
        return self._watermark

    def append(self, batch: AppEventBatch) -> None:
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
        self._ids_by_event_time.update(staged)
        self._events += len(batch.event_id)
        self._batches.append(batch)
        if len(batch.event_time):
            self._watermark = max(
                self._watermark, int(batch.event_time.max())
            )

    def read(self, *, through: int | None = None) -> AppEventBatch:
        result = AppEventBatch.concatenate(self._batches)
        if through is None:
            return result
        return result.select(result.event_time <= through)

    def manifest(self) -> dict[str, int | str]:
        return {
            "schema": "observable-app-events-v2",
            "events": self._events,
            "batches": len(self._batches),
            "watermark": self._watermark,
        }
