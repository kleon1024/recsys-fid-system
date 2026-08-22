"""Extract the POI vertical from main Feed impressions and close delayed labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .contracts import (
    ACTION_WINDOWS_SECONDS,
    FeedAction,
    FeedImpression,
    PoiFeedExample,
)
from .streaming import ViewerFeatureOperator


@dataclass(frozen=True)
class PoiFeedJoinReport:
    examples: tuple[PoiFeedExample, ...]
    main_impressions: int
    anchored_impressions: int
    immature: int
    duplicate_actions: int
    ignored_actions: int


class PoiFeedJoiner:
    def __init__(self, allowed_lateness_seconds: int = 300) -> None:
        self.allowed_lateness_seconds = allowed_lateness_seconds

    def build(
        self,
        impressions: list[FeedImpression],
        actions: list[FeedAction],
        features: ViewerFeatureOperator,
        watermark: int,
    ) -> PoiFeedJoinReport:
        anchored = [value for value in impressions if value.poi_id is not None]
        action_by_id: dict[str, FeedAction] = {}
        duplicates = 0
        for action in actions:
            duplicates += int(action.action_id in action_by_id)
            action_by_id.setdefault(action.action_id, action)
        actions_by_impression: dict[str, list[FeedAction]] = defaultdict(list)
        impression_ids = {value.impression_id for value in anchored}
        ignored = 0
        for action in action_by_id.values():
            if action.impression_id not in impression_ids:
                ignored += 1
                continue
            actions_by_impression[action.impression_id].append(action)
        examples: list[PoiFeedExample] = []
        immature = 0
        max_window = max(ACTION_WINDOWS_SECONDS.values())
        for impression in anchored:
            if watermark < impression.event_time + max_window + self.allowed_lateness_seconds:
                immature += 1
                continue
            valid = actions_by_impression[impression.impression_id]
            labels = {
                task: float(
                    any(
                        action.action == task
                        and impression.event_time <= action.event_time
                        <= impression.event_time + window
                        and action.received_at <= action.event_time + self.allowed_lateness_seconds
                        for action in valid
                    )
                )
                for task, window in ACTION_WINDOWS_SECONDS.items()
            }
            snapshot = features.snapshot(impression.viewer_id, impression.event_time)
            dense = list(impression.base_features)
            dense[5] = min(snapshot.count_7d / 50.0, 1.0)
            dense[6] = float(
                impression.category_id in snapshot.category_sequence[-8:]
            )
            examples.append(
                PoiFeedExample(
                    impression_id=impression.impression_id,
                    viewer_id=impression.viewer_id,
                    author_id=impression.author_id,
                    video_id=impression.video_id,
                    poi_id=int(impression.poi_id),
                    event_time=impression.event_time,
                    features=tuple(dense),
                    sequence=snapshot.sequence_tensor(),
                    labels=labels,
                    media_version=impression.media_version,
                    feature_version=impression.feature_version,
                    model_version=impression.model_version,
                    index_version=impression.index_version,
                )
            )
        return PoiFeedJoinReport(
            tuple(examples), len(impressions), len(anchored), immature, duplicates, ignored
        )
