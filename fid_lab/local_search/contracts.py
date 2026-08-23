"""Local Search corpus, request, label, and GPU scale contracts."""

from __future__ import annotations

from dataclasses import dataclass


LOCAL_SEARCH_ROUTES = (
    "lexical", "geo", "semantic_tower", "history", "retarget"
)
LOCAL_SEARCH_TASKS = ("click", "detail", "save", "order")


@dataclass(frozen=True)
class LocalSearchConfig:
    requests: int = 120_000
    pois: int = 16_384
    users: int = 60_000
    categories: int = 32
    cities: int = 64
    semantic_dim: int = 16
    history_length: int = 24
    route_candidates: int = 12
    merged_candidates: int = 24
    exposed_candidates: int = 8
    train_epochs: int = 3
    train_batch_requests: int = 1_024
    learning_rate: float = 1.5e-3
    seed: int = 20260824
    device: str = "cuda:0"

    def __post_init__(self):
        if self.requests < 400:
            raise ValueError("Local Search world requires at least 400 requests")
        if self.users > self.requests:
            raise ValueError("Local Search users cannot exceed requests")
        if self.pois % self.categories:
            raise ValueError("POI corpus must partition by category")
        if self.semantic_dim % 4:
            raise ValueError("semantic width must support four attention heads")
        if self.merged_candidates > 2 * self.route_candidates:
            raise ValueError("baseline lexical and geo routes cannot fill the pool")
        if self.exposed_candidates > self.merged_candidates:
            raise ValueError("exposure count exceeds the merged candidate pool")
