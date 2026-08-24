"""Point-in-time feature updates compiled only from observable app events."""

from __future__ import annotations

import torch

from ..contracts import Surface
from ..exchange import ObservableResponse
from .state import CatalogState, UserState


def apply_response_events(
    users: UserState,
    catalog: CatalogState,
    response: ObservableResponse,
    surface: torch.Tensor,
) -> None:
    """Update online estimates without access to any hidden state."""
    item = response.selected_item
    positive = (
        response.event("long_view") | response.event("like")
        | response.event("click") | response.event("order")
        | response.event("publish")
    )
    negative = response.event("negative")
    embedding = catalog.topic_embedding[item]
    active = response.active[:, None]
    rate = (0.03 + 0.12 * positive.float())[:, None]
    updated_short = torch.nn.functional.normalize(
        (1.0 - rate) * users.short_interest + rate * embedding, dim=1
    )
    users.short_interest = torch.where(
        active, updated_short, users.short_interest
    )
    updated_observed = torch.nn.functional.normalize(
        0.97 * users.observed_interest + 0.03 * updated_short, dim=1
    )
    users.observed_interest = torch.where(
        active, updated_observed, users.observed_interest
    )
    observed_reward = (
        0.015 * torch.log1p(response.stay_seconds)
        + 0.035 * positive.float() - 0.10 * negative.float()
    )
    users.satisfaction_estimate = torch.where(
        response.active,
        (
            0.97 * users.satisfaction_estimate + observed_reward
        ).clamp(0.0, 1.0),
        users.satisfaction_estimate,
    )
    users.fatigue_counter = torch.where(
        response.active,
        (
            0.90 * users.fatigue_counter + 0.02
            + 0.04 * negative.float()
            + 0.008 * users.session_depth.float()
        ).clamp(0.0, 1.0),
        users.fatigue_counter,
    )
    users.commerce_intent_estimate = (
        0.97 * users.commerce_intent_estimate
        + 0.08 * response.event("add_cart").float()
        + 0.12 * response.event("order").float()
    ).clamp(0.0, 1.0)
    users.local_intent_estimate = (
        0.98 * users.local_intent_estimate
        + 0.08 * ((surface == int(Surface.LOCAL)) & positive).float()
    ).clamp(0.0, 1.0)
    users.creator_intent_estimate = (
        0.98 * users.creator_intent_estimate
        + 0.15 * response.event("publish").float()
    ).clamp(0.0, 1.0)
    users.activity_rate_estimate = (
        0.99 * users.activity_rate_estimate
        + 0.01 * response.active.float()
    ).clamp(0.02, 0.95)
    users.trend_affinity_estimate = (
        0.995 * users.trend_affinity_estimate
        + 0.005 * positive.float()
    ).clamp(0.0, 1.0)
    surface_signal = torch.nn.functional.one_hot(
        surface, users.surface_affinity_estimate.shape[1]
    ).float() * (0.25 + 0.75 * positive.float())[:, None]
    users.surface_affinity_estimate = (
        0.99 * users.surface_affinity_estimate + 0.01 * surface_signal
    ).clamp(0.0, 1.0)
    users.cold_start_confidence = (
        users.cold_start_confidence
        + response.active.float() * (0.015 + 0.035 * positive.float())
    ).clamp(0.0, 1.0)
    users.session_depth += response.active.long()
    users.request_index += response.active.long()


def apply_daily_observations(users: UserState) -> None:
    """Update account-age and activity features from the closed platform day."""
    users.tenure_days += users.registered.long()
    users.activity_rate_estimate = (
        0.98 * users.activity_rate_estimate + 0.02 * users.active.float()
    ).clamp(0.02, 0.95)
