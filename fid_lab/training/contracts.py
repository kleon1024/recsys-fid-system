"""Immutable event, example, prediction, and artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TASKS = ("click", "like", "long_view")


@dataclass(frozen=True)
class ImpressionEvent:
    request_id: str
    user_id: int
    item_id: int
    event_time: int
    position: int
    propensity: float
    feature_fids: tuple[int, ...]
    feature_buckets: tuple[int, ...]
    schema_version: str
    served_model_version: int

    @property
    def key(self) -> tuple[str, int]:
        return self.request_id, self.item_id


@dataclass(frozen=True)
class ActionEvent:
    event_id: str
    request_id: str
    item_id: int
    action: str
    event_time: int
    received_at: int
    value: float = 1.0

    @property
    def key(self) -> tuple[str, int]:
        return self.request_id, self.item_id


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    user_id: int
    item_id: int
    impression_time: int
    feature_fids: tuple[int, ...]
    feature_buckets: tuple[int, ...]
    labels: Mapping[str, float]
    sample_weight: float
    schema_version: str


@dataclass(frozen=True)
class PredictionRecord:
    user_id: int
    label: float
    score: float
    model_version: int
    slice_name: str = "all"


@dataclass(frozen=True)
class ChainManifest:
    schema_version: str
    fid_layout: str
    joiner_version: str
    model_version: int
    vector_index_version: str
    tasks: tuple[str, ...] = TASKS
