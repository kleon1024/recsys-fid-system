"""Contracts for the request-level POI posting world."""

from __future__ import annotations

from dataclasses import dataclass


POSTING_ROUTES = ("popular", "geo", "semantic", "history")
POSTING_TASKS = ("select", "publish", "relevance")


@dataclass(frozen=True)
class PostingWorldConfig:
    requests: int = 200_000
    cities: int = 64
    categories: int = 16
    items_per_cell: int = 64
    semantic_dim: int = 16
    route_candidates: int = 12
    merged_candidates: int = 20
    exposed_candidates: int = 8
    batch_requests: int = 50_000
    train_epochs: int = 3
    train_batch_pairs: int = 16_384
    learning_rate: float = 2e-3
    seed: int = 20260824
    device: str = "cuda:0"

    @property
    def items(self):
        return self.cities * self.categories * self.items_per_cell

    def __post_init__(self):
        if self.merged_candidates > 2 * self.route_candidates:
            raise ValueError("base posting routes cannot fill the merged pool")
        if self.exposed_candidates > self.merged_candidates:
            raise ValueError("posting exposure cannot exceed merged candidates")
        if self.semantic_dim < 4 or self.requests < 100:
            raise ValueError("posting world is too small")
