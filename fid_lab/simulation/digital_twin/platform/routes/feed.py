"""Observable Feed route signals shared by route-specific candidate builders."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...catalog import PublicCatalog
from ...contracts import EventType
from ..lifecycle import ContentLifecycle
from ..projection import ITEM_COUNTER_EVENTS, PlatformProjectionState


MAIN_FEED_LIFECYCLES = (
    ContentLifecycle.COLD_START,
    ContentLifecycle.RECENT,
    ContentLifecycle.HOT,
)


@dataclass(frozen=True)
class FeedRouteSignals:
    engagement_rate: torch.Tensor
    random: torch.Tensor
    popular: torch.Tensor
    cold_start: torch.Tensor
    hot: torch.Tensor
    evergreen: torch.Tensor
    following: torch.Tensor


def build_feed_route_signals(
    catalog: PublicCatalog,
    state: PlatformProjectionState,
    current_time: torch.Tensor,
) -> FeedRouteSignals:
    engagement_rate = state.item_recent_engagements / (
        state.item_recent_impressions.clamp_min(1.0)
    )
    impression = state.item_event_counts[
        :, ITEM_COUNTER_EVENTS.index(EventType.IMPRESSION)
    ]
    negative = state.item_event_counts[
        :, ITEM_COUNTER_EVENTS.index(EventType.NEGATIVE)
    ]
    smoothed_engagement = (
        state.item_recent_engagements + 1.0
    ) / (state.item_recent_impressions + 20.0)
    smoothed_negative = (negative + 1.0) / (impression + 50.0)
    age = (
        current_time - state.item_publish_time.clamp_max(current_time)
    ).clamp_min(0).float()
    return FeedRouteSignals(
        engagement_rate=engagement_rate,
        random=torch.zeros_like(catalog.quality_prior),
        popular=(
            torch.log1p(state.item_recent_impressions)
            * smoothed_engagement
            - 0.50 * smoothed_negative
        ),
        cold_start=(
            0.65 * catalog.quality_prior
            - 0.08 * torch.log1p(state.item_recent_impressions)
            - 0.0005 * age
        ),
        hot=(
            0.55 * engagement_rate
            + 0.22 * torch.log1p(state.item_recent_impressions)
            + 0.23 * catalog.quality_prior
        ),
        evergreen=(
            0.72 * catalog.quality_prior + 0.28 * engagement_rate
        ),
        following=catalog.quality_prior + 0.25 * engagement_rate,
    )
