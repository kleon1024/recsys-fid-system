"""One authority for stage logs, delayed labels, and training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TASK_WINDOWS_SECONDS = {
    "play": 300,
    "play_3s": 300,
    "stay_seconds": 300,
    "slide": 300,
    "long_view": 300,
    "high_quality_long_view": 300,
    "like": 600,
    "comment": 600,
    "share": 600,
    "negative_feedback": 600,
    "anchor_impression": 600,
    "anchor_click": 600,
    "poi_detail": 1_800,
    "poi_favorite": 86_400,
    "favorite": 86_400,
    "submit": 86_400,
    "order": 604_800,
    "payment": 604_800,
    "pixel_conversion": 604_800,
}

BEHAVIOR_STRENGTH = {
    "play": 0.25,
    "play_3s": 0.5,
    "stay_seconds": 0.75,
    "slide": 0.0,
    "long_view": 1.0,
    "high_quality_long_view": 1.5,
    "like": 2.0,
    "comment": 2.5,
    "share": 3.0,
    "negative_feedback": 0.0,
    "anchor_impression": 0.0,
    "anchor_click": 2.0,
    "poi_detail": 3.0,
    "poi_favorite": 4.0,
    "favorite": 4.0,
    "submit": 5.0,
    "order": 6.0,
    "payment": 7.0,
    "pixel_conversion": 7.0,
}


@dataclass(frozen=True)
class StageDecision:
    request_id: str
    viewer_id: int
    author_id: int
    video_id: int
    poi_id: int
    impression_time: int
    category_id: int
    city_id: int
    feature_fids: tuple[int, ...]
    dense_features: tuple[float, ...]
    sequence: tuple[tuple[float, ...], ...]
    recall_route: str
    sampling_probability: float
    teacher_score: float
    teacher_rank: int
    exposed: bool
    pixel_observable: bool
    served_scores: Mapping[str, float]
    manifest: Mapping[str, str]

    @property
    def key(self) -> tuple[str, int, int]:
        return self.request_id, self.video_id, self.poi_id


@dataclass(frozen=True)
class ActionEvent:
    event_id: str
    request_id: str
    video_id: int
    poi_id: int
    action: str
    event_time: int
    received_at: int
    value: float = 1.0

    @property
    def key(self) -> tuple[str, int, int]:
        return self.request_id, self.video_id, self.poi_id


@dataclass(frozen=True)
class CommerceEvent:
    event_id: str
    request_id: str
    video_id: int
    poi_id: int
    action: str
    order_id: str
    payment_id: str | None
    event_time: int
    received_at: int

    @property
    def key(self) -> tuple[str, int, int]:
        return self.request_id, self.video_id, self.poi_id


@dataclass(frozen=True)
class OutboundClick:
    click_id: str
    request_id: str
    video_id: int
    poi_id: int
    identity: str | None
    merchant_id: int
    event_time: int

    @property
    def key(self) -> tuple[str, int, int]:
        return self.request_id, self.video_id, self.poi_id


@dataclass(frozen=True)
class PixelEvent:
    event_id: str
    conversion_id: str
    identity: str | None
    merchant_id: int
    event_time: int
    received_at: int
    click_id: str | None = None


@dataclass(frozen=True)
class NegativeSample:
    item_id: int
    source: str
    sampling_probability: float


@dataclass(frozen=True)
class RecallExample:
    request_id: str
    viewer_id: int
    positive_item_id: int
    behavior_strength: float
    negatives: tuple[NegativeSample, ...]
    manifest: Mapping[str, str]


@dataclass(frozen=True)
class CoarseRankExample:
    key: tuple[str, int, int]
    feature_fids: tuple[int, ...]
    dense_features: tuple[float, ...]
    hard_labels: Mapping[str, float]
    label_masks: Mapping[str, bool]
    teacher_score: float
    teacher_rank: int
    recall_route: str
    sampling_probability: float
    served_scores: Mapping[str, float]
    manifest: Mapping[str, str]


@dataclass(frozen=True)
class FineRankExample:
    key: tuple[str, int, int]
    viewer_id: int
    author_id: int
    feature_fids: tuple[int, ...]
    dense_features: tuple[float, ...]
    sequence: tuple[tuple[float, ...], ...]
    labels: Mapping[str, float]
    label_masks: Mapping[str, bool]
    sample_weight: float
    served_scores: Mapping[str, float]
    manifest: Mapping[str, str]
