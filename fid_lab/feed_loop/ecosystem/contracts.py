"""Contracts for cross-day Feed and creator-supply rollouts."""

from __future__ import annotations

from dataclasses import dataclass


CONSUMER_GUARDRAILS = {
    "creator_posts_absolute": -0.002,
    "creator_active_absolute": -0.001,
    "negative_events_per_user": 0.01,
}
CREATOR_RETENTION_GUARDRAILS = {
    "stay_seconds_per_user": -0.10,
    "lt_per_user": -0.005,
    "creator_posts_absolute": -0.0001,
    "negative_events_per_user": 0.01,
}


@dataclass(frozen=True)
class EcosystemConfig:
    days: int = 7
    steps_per_day: int = 8
    max_new_items_per_day: int = 5_000
    seed: int = 20260824
    objective: str = "consumer"

    def __post_init__(self):
        if self.days < 2 or self.steps_per_day < 2:
            raise ValueError("ecosystem rollout requires multiple days and requests")
        if self.max_new_items_per_day < 1:
            raise ValueError("daily supply budget must be positive")
        if self.objective not in {"consumer", "creator_retention"}:
            raise ValueError(
                "ecosystem objective must be consumer or creator_retention"
            )
