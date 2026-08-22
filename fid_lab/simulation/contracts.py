"""Contracts for the stateful behavior environment and policy traces."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Catalog:
    topics: np.ndarray
    quality: np.ndarray
    commerce_value: np.ndarray
    category: np.ndarray
    city: np.ndarray
    author: np.ndarray
    poi: np.ndarray


@dataclass(frozen=True)
class Response:
    long_view: bool
    anchor_click: bool
    detail: bool
    favorite: bool
    order: bool
    payment: bool
    pixel_conversion: bool
    negative_feedback: bool
    watch_minutes: float
    probabilities: tuple[float, ...]


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
    features: tuple[float, ...]
    score: float
    response: Response
    returned_next_session: bool


@dataclass(frozen=True)
class Trajectory:
    rows: tuple[TraceRow, ...]
    sessions: int
    returned_sessions: int
    watch_minutes: float
    anchor_clicks: int
    orders: int
    negative_feedback: int
    discounted_value: float
