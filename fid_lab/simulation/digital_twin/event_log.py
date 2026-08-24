"""Observable append-only event authority with idempotency and watermark."""

from __future__ import annotations

from .contracts import AppEventBatch


class ObservableEventLog:
    def __init__(self):
        self._batches: list[AppEventBatch] = []
        self._event_ids: set[int] = set()
        self._watermark = -1

    @property
    def watermark(self) -> int:
        return self._watermark

    def append(self, batch: AppEventBatch) -> None:
        ids = set(int(value) for value in batch.event_id.cpu())
        duplicate = ids & self._event_ids
        if duplicate:
            raise ValueError(f"event log duplicate ids: {len(duplicate)}")
        self._event_ids.update(ids)
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
            "schema": "observable-app-events-v1",
            "events": len(self._event_ids),
            "batches": len(self._batches),
            "watermark": self._watermark,
        }
