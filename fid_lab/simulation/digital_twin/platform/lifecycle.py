"""Observable content lifecycle authority for Feed corpus membership."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from ..contracts import ContentKind


LIFECYCLE_POLICY_VERSION = "content-lifecycle-v1"


class ContentLifecycle(IntEnum):
    RESERVED = 0
    COLD_START = 1
    RECENT = 2
    HOT = 3
    EVERGREEN = 4
    EXPIRED = 5


@dataclass(frozen=True)
class LifecycleConfig:
    ticks_per_day: int = 96
    recent_days: int = 30
    cold_start_days: int = 2
    hot_half_life_days: float = 1.0
    hot_min_impressions: float = 12.0
    hot_engagement_rate: float = 0.18
    evergreen_min_impressions: float = 40.0
    evergreen_engagement_rate: float = 0.12

    def __post_init__(self):
        if self.ticks_per_day <= 0 or self.recent_days <= 0:
            raise ValueError("lifecycle time dimensions must be positive")
        if not 0 < self.cold_start_days < self.recent_days:
            raise ValueError("cold start must be inside recent window")
        if self.hot_half_life_days <= 0:
            raise ValueError("hot half life must be positive")

    @property
    def recent_ticks(self) -> int:
        return self.recent_days * self.ticks_per_day

    @property
    def cold_start_ticks(self) -> int:
        return self.cold_start_days * self.ticks_per_day

    @property
    def hot_half_life_ticks(self) -> float:
        return self.hot_half_life_days * self.ticks_per_day


def post_content_mask(content_kind: torch.Tensor) -> torch.Tensor:
    return (
        (content_kind == int(ContentKind.SHORT_VIDEO))
        | (content_kind == int(ContentKind.PHOTO))
        | (content_kind == int(ContentKind.ARTICLE))
        | (content_kind == int(ContentKind.CARD))
    )


def classify_lifecycle(
    *,
    active: torch.Tensor,
    content_kind: torch.Tensor,
    publish_time: torch.Tensor,
    evergreen_eligible: torch.Tensor,
    recent_impressions: torch.Tensor,
    recent_engagements: torch.Tensor,
    logical_time: int,
    config: LifecycleConfig,
) -> torch.Tensor:
    """Classify from observable age, eligibility and decayed feedback."""
    lifecycle = torch.full_like(
        publish_time, int(ContentLifecycle.RESERVED),
    )
    post = post_content_mask(content_kind)
    business_entity = active & ~post
    lifecycle[business_entity] = int(ContentLifecycle.EVERGREEN)
    age = (
        logical_time - publish_time.clamp_max(logical_time)
    ).clamp_min(0)
    impressions = recent_impressions.clamp_min(0.0)
    rate = recent_engagements / impressions.clamp_min(1.0)
    cold = active & post & (age < config.cold_start_ticks)
    recent = active & post & (age <= config.recent_ticks) & ~cold
    hot = (
        recent
        & (impressions >= config.hot_min_impressions)
        & (rate >= config.hot_engagement_rate)
    )
    evergreen = (
        active
        & post
        & (age > config.recent_ticks)
        & (
            evergreen_eligible
            | (
                (impressions >= config.evergreen_min_impressions)
                & (rate >= config.evergreen_engagement_rate)
            )
        )
    )
    lifecycle[cold] = int(ContentLifecycle.COLD_START)
    lifecycle[recent] = int(ContentLifecycle.RECENT)
    lifecycle[hot] = int(ContentLifecycle.HOT)
    lifecycle[active & post & (age > config.recent_ticks)] = int(
        ContentLifecycle.EXPIRED
    )
    lifecycle[evergreen] = int(ContentLifecycle.EVERGREEN)
    return lifecycle
