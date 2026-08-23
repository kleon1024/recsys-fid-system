"""Contracts for the stateful behavior environment and policy traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    users: int = 2_000
    items: int = 4_000
    topics: int = 12
    candidates: int = 20
    max_sessions: int = 4
    requests_per_session: int = 8
    exploration_rate: float = 0.08
    joiner_users: int = 100
    seed: int = 20260823
    signal_version: str = "industrial-cross-sequence-v1"

    def __post_init__(self) -> None:
        if self.signal_version not in {
            "industrial-cross-sequence-v1",
            "heterogeneous-nonlinear-v2",
        }:
            raise ValueError(f"unsupported signal version: {self.signal_version}")


@dataclass(frozen=True)
class Catalog:
    topics: np.ndarray
    quality: np.ndarray
    commerce_value: np.ndarray
    popularity: np.ndarray
    freshness: np.ndarray
    duration_seconds: np.ndarray
    is_poi_video: np.ndarray
    category: np.ndarray
    city: np.ndarray
    author: np.ndarray
    poi: np.ndarray
    poi_quality: np.ndarray
    inventory_available: np.ndarray
    fulfillment: np.ndarray


@dataclass(frozen=True)
class Response:
    play: bool
    play_3s: bool
    stay_seconds: float
    slide: bool
    long_view: bool
    high_quality_long_view: bool
    like: bool
    favorite: bool
    comment: bool
    share: bool
    anchor_impression: bool
    anchor_click: bool
    poi_detail: bool
    poi_favorite: bool
    order: bool
    payment: bool
    pixel_conversion: bool
    negative_feedback: bool
    watch_minutes: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class TraceRow:
    user_id: int
    session_id: int
    request_index: int
    request_id: str
    item_id: int
    candidate_ids: tuple[int, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    candidate_scores: tuple[float, ...]
    candidate_propensities: tuple[float, ...]
    candidate_oracle_long_view: tuple[float, ...]
    candidate_routes: tuple[tuple[str, ...], ...]
    recall_count: int
    coarse_count: int
    features: tuple[float, ...]
    score: float
    selection_probability: float
    response: Response
    returned_next_session: bool
    parameter_snapshot: Mapping[str, object] | None = None
    query_embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class Trajectory:
    rows: tuple[TraceRow, ...]
    sessions: int
    returned_sessions: int
    plays: int
    play_3s: int
    stay_seconds: float
    slides: int
    long_views: int
    high_quality_long_views: int
    likes: int
    favorites: int
    comments: int
    shares: int
    watch_minutes: float
    anchor_impressions: int
    anchor_clicks: int
    poi_details: int
    poi_favorites: int
    orders: int
    negative_feedback: int
    lt_value: float
    local_value_tree_score: float
    lt_components: Mapping[str, float]
    business_value_components: Mapping[str, float]


@dataclass(frozen=True)
class PostingResponse:
    entered_posting_page: bool
    poi_candidates_shown: int
    poi_selected: bool
    submitted: bool
    published: bool
    selected_poi_id: int | None
    predicted_content_quality: float
