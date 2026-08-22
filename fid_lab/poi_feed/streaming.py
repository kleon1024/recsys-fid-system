"""Flink-compatible keyed-state semantics for realtime and sequence features."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .contracts import ViewerBehaviorEvent, ViewerFeatureSnapshot


ACTION_IDS = {
    "view": 0,
    "long_view": 1,
    "anchor_click": 2,
    "favorite": 3,
    "order": 4,
}


@dataclass(frozen=True)
class StreamReport:
    accepted: int
    late: int
    duplicates: int


class ViewerFeatureOperator:
    """Reference operator: keyBy(viewer), event time, watermark, and bounded state."""

    def __init__(self, allowed_lateness_seconds: int = 300) -> None:
        self.allowed_lateness_seconds = allowed_lateness_seconds
        self.events: dict[int, list[ViewerBehaviorEvent]] = defaultdict(list)
        self.event_ids: set[str] = set()
        self.accepted = 0
        self.late = 0
        self.duplicates = 0

    def ingest(self, event: ViewerBehaviorEvent, watermark: int) -> None:
        if event.event_id in self.event_ids:
            self.duplicates += 1
            return
        self.event_ids.add(event.event_id)
        if event.event_time < watermark - self.allowed_lateness_seconds:
            self.late += 1
            return
        self.events[event.viewer_id].append(event)
        self.events[event.viewer_id].sort(key=lambda value: (value.event_time, value.event_id))
        self.accepted += 1

    def snapshot(self, viewer_id: int, as_of: int) -> ViewerFeatureSnapshot:
        history = [
            event for event in self.events.get(viewer_id, ()) if event.event_time <= as_of
        ]
        recent = history[-64:]
        return ViewerFeatureSnapshot(
            viewer_id=viewer_id,
            as_of=as_of,
            count_1h=sum(event.event_time >= as_of - 3_600 for event in history),
            count_7d=sum(event.event_time >= as_of - 604_800 for event in history),
            category_sequence=tuple(event.category_id for event in recent),
            action_sequence=tuple(ACTION_IDS.get(event.action, 0) for event in recent),
        )

    def report(self) -> StreamReport:
        return StreamReport(self.accepted, self.late, self.duplicates)
