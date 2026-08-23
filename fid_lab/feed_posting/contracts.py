"""Contracts for Feed posting prompts, labels, and GPU scale."""

from __future__ import annotations

from dataclasses import dataclass


FEED_POSTING_ROUTES = ("trending", "i2i", "creator_history", "semantic")
FEED_POSTING_TASKS = ("click", "create", "publish", "quality")


@dataclass(frozen=True)
class FeedPostingConfig:
    requests: int = 150_000
    prompts: int = 32_768
    categories: int = 32
    countries: int = 12
    semantic_dim: int = 16
    sequence_length: int = 24
    route_candidates: int = 12
    merged_candidates: int = 20
    exposed_candidates: int = 6
    train_epochs: int = 3
    train_batch_requests: int = 1_024
    learning_rate: float = 1.5e-3
    seed: int = 20260824
    device: str = "cuda:0"

    def __post_init__(self):
        if self.merged_candidates > 2 * self.route_candidates:
            raise ValueError("base Feed-posting routes cannot fill candidate pool")
        if self.exposed_candidates > self.merged_candidates:
            raise ValueError("Feed-posting exposure exceeds candidate pool")
        if self.semantic_dim % 4:
            raise ValueError("Feed-posting semantic width must support four heads")
        if self.prompts % self.categories:
            raise ValueError("Feed-posting prompts must partition by category")
        if self.requests < 100:
            raise ValueError("Feed-posting world is too small")
