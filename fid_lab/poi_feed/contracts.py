"""Event and example authority for POI-anchored Feed distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_WINDOWS_SECONDS = {
    "long_view": 300,
    "anchor_click": 600,
    "detail_view": 1_800,
    "favorite": 86_400,
    "order": 604_800,
    "negative_feedback": 3_600,
}


@dataclass(frozen=True)
class ViewerBehaviorEvent:
    event_id: str
    viewer_id: int
    category_id: int
    action: str
    event_time: int
    received_at: int


@dataclass(frozen=True)
class FeedImpression:
    impression_id: str
    viewer_id: int
    author_id: int
    video_id: int
    poi_id: int | None
    category_id: int
    event_time: int
    base_features: tuple[float, ...]
    media_version: str
    feature_version: str
    model_version: str
    index_version: str


@dataclass(frozen=True)
class FeedAction:
    action_id: str
    impression_id: str
    action: str
    event_time: int
    received_at: int


@dataclass(frozen=True)
class ViewerFeatureSnapshot:
    viewer_id: int
    as_of: int
    count_1h: int
    count_7d: int
    category_sequence: tuple[int, ...]
    action_sequence: tuple[int, ...]

    def sequence_tensor(self, length: int = 24) -> np.ndarray:
        result = np.zeros((length, 8), dtype=np.float32)
        pairs = list(zip(self.category_sequence, self.action_sequence))[-length:]
        start = length - len(pairs)
        for offset, (category, action) in enumerate(pairs, start=start):
            result[offset, category % 6] = 1.0
            result[offset, 6] = action / 4.0
            result[offset, 7] = 1.0
        return result


@dataclass(frozen=True)
class PoiFeedExample:
    impression_id: str
    viewer_id: int
    author_id: int
    video_id: int
    poi_id: int
    event_time: int
    features: tuple[float, ...]
    sequence: np.ndarray
    labels: dict[str, float]
    media_version: str
    feature_version: str
    model_version: str
    index_version: str
