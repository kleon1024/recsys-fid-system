"""Authority for the production-like synthetic Feed distribution."""

from __future__ import annotations

from dataclasses import dataclass


FEED_TASKS = (
    "long_view",
    "anchor_click",
    "detail_view",
    "favorite",
    "order",
    "negative_feedback",
)


@dataclass(frozen=True)
class ScaleConfig:
    """A replaceable scenario, not a claim about any private production rate."""

    main_impressions: int = 1_000_000
    viewers: int = 100_000
    authors: int = 20_000
    videos: int = 500_000
    pois: int = 50_000
    anchor_rate: float = 0.02
    long_view_rate: float = 0.28
    anchor_click_rate: float = 0.022
    detail_given_click: float = 0.34
    favorite_given_detail: float = 0.24
    order_given_detail: float = 0.08
    negative_feedback_rate: float = 0.006
    seed: int = 20260822
    signal_version: str = "industrial-cross-sequence-v1"

    def __post_init__(self) -> None:
        counts = (self.main_impressions, self.viewers, self.authors, self.videos, self.pois)
        if min(counts) <= 0:
            raise ValueError("scale counts must be positive")
        rates = (
            self.anchor_rate,
            self.long_view_rate,
            self.anchor_click_rate,
            self.detail_given_click,
            self.favorite_given_detail,
            self.order_given_detail,
            self.negative_feedback_rate,
        )
        if any(rate < 0.0 or rate > 1.0 for rate in rates):
            raise ValueError("rates must be probabilities")
