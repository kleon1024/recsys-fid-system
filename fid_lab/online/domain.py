"""Shared request, candidate, and trace contracts for the recommendation chain."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np


@dataclass(frozen=True, eq=False)
class Item:
    item_id: int
    content_type: str
    category: str
    creator_id: int
    embedding: np.ndarray
    popularity: float
    quality: float
    age_hours: float
    allowed_countries: frozenset[str]
    is_safe: bool = True
    is_active: bool = True


@dataclass(frozen=True, eq=False)
class RequestContext:
    request_id: str
    user_id: int
    country: str
    user_embedding: np.ndarray
    category_affinity: Mapping[str, float]
    device: int = 0
    hour_bucket: int = 0
    seen_item_ids: frozenset[int] = frozenset()
    pinned_item_id: int | None = None
    size: int = 20


@dataclass(frozen=True)
class Candidate:
    item_id: int
    recall_scores: Mapping[str, float] = field(default_factory=dict)
    recall_reasons: tuple[str, ...] = ()
    feature_fids: tuple[int, ...] = ()
    feature_buckets: tuple[int, ...] = ()
    coarse_score: float = 0.0
    predictions: Mapping[str, float] = field(default_factory=dict)
    value_score: float = 0.0
    rule_score: float = 0.0
    final_score: float = 0.0

    def update(self, **changes: object) -> "Candidate":
        return replace(self, **changes)


@dataclass(frozen=True)
class StageTrace:
    stage: str
    input_count: int
    output_count: int
    latency_ms: float
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationResult:
    request_id: str
    items: tuple[Candidate, ...]
    traces: tuple[StageTrace, ...]
    artifact_versions: Mapping[str, str]
