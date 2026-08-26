"""Observable country and recent-interest Popular retrieval routes."""

from __future__ import annotations

import torch

from ...catalog import PublicCatalog
from ...contracts import EventType, PlatformRequestBatch, Surface
from ..indexes.contracts import RetrievalConfig
from ..projection import PlatformProjectionState
from .contracts import surface_eligibility
from .feed import MAIN_FEED_LIFECYCLES


def _eligible_feed_items(
    catalog: PublicCatalog,
    state: PlatformProjectionState,
) -> torch.Tensor:
    lifecycle = torch.zeros_like(state.item_active)
    for value in MAIN_FEED_LIFECYCLES:
        lifecycle |= state.item_lifecycle == int(value)
    return (
        state.item_active
        & lifecycle
        & surface_eligibility(int(Surface.FEED), catalog.content_kind)
    )


def popular_candidates(
    catalog: PublicCatalog,
    config: RetrievalConfig,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rotating country-popular pools learned from mature events."""
    rows, limit = len(requests.user_id), config.route_k
    item = torch.full(
        (rows, limit), -1, device=catalog.item_id.device, dtype=torch.long,
    )
    values = torch.full_like(item, -torch.inf, dtype=torch.float)
    feed = requests.surface == int(Surface.FEED)
    request_country = state.user_country[requests.user_id]
    eligible_base = _eligible_feed_items(catalog, state)
    for country in torch.unique(request_country[feed]).tolist():
        selected = feed & (request_country == country)
        candidates = torch.where(
            eligible_base & (state.item_country == country),
        )[0]
        pool_width = min(
            len(candidates), limit * config.popular_pool_multiplier,
        )
        if not pool_width:
            continue
        pool = candidates[torch.topk(score[candidates], pool_width).indices]
        target = torch.where(selected)[0]
        width = min(limit, pool_width)
        start = torch.remainder(
            requests.request_id[target] * 503
            + requests.user_id[target] * 1_009,
            pool_width,
        )
        offset = torch.arange(width, device=catalog.item_id.device)[None]
        chosen = pool[torch.remainder(start[:, None] + offset, pool_width)]
        item[target, :width] = chosen
        values[target, :width] = score[chosen]
    return item, values


def _recent_interest_topic(
    catalog: PublicCatalog,
    config: RetrievalConfig,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
) -> torch.Tensor:
    user_id = requests.user_id
    history_item = state.user_history_item[user_id]
    event_type = state.user_history_event_type[user_id]
    event_time = state.user_history_event_time[user_id]
    strength = torch.zeros_like(event_time, dtype=torch.float)
    for candidate, value in (
        (EventType.PLAY_3S, 1.0),
        (EventType.LONG_VIEW, 2.0),
        (EventType.COMPLETE, 2.5),
        (EventType.CLICK, 2.0),
        (EventType.LIKE, 3.0),
        (EventType.FAVORITE, 3.5),
        (EventType.SHARE, 4.0),
    ):
        strength = torch.where(
            event_type == int(candidate), torch.full_like(strength, value),
            strength,
        )
    positive = strength > 0.0
    valid = positive & (history_item >= 0)
    age = (
        requests.event_time[:, None] - event_time
    ).clamp_min(0).float()
    weight = strength * torch.exp2(
        -age / float(config.interest_half_life_ticks),
    )
    topic = catalog.topic_id[history_item.clamp_min(0)]
    topic_score = torch.zeros(
        len(requests.user_id),
        int(catalog.topic_id.max()) + 1,
        device=catalog.item_id.device,
    )
    topic_score.scatter_add_(1, topic, weight * valid)
    selected_topic = topic_score.argmax(dim=1)
    return torch.where(
        valid.any(dim=1), selected_topic, torch.full_like(selected_topic, -1),
    )


def interest_popular_candidates(
    catalog: PublicCatalog,
    config: RetrievalConfig,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    score: torch.Tensor,
    *,
    interest_fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve popular items inside the latest observable interest segment."""
    item, values = popular_candidates(
        catalog, config, requests, state, score,
    )
    feed = requests.surface == int(Surface.FEED)
    country = state.user_country[requests.user_id]
    topic = _recent_interest_topic(catalog, config, requests, state)
    eligible_base = _eligible_feed_items(catalog, state)
    segment = country * (int(catalog.topic_id.max()) + 1) + topic
    for key in torch.unique(segment[feed & (topic >= 0)]).tolist():
        selected = feed & (segment == key)
        target = torch.where(selected)[0]
        selected_country = int(country[target[0]])
        selected_topic = int(topic[target[0]])
        candidates = torch.where(
            eligible_base
            & (state.item_country == selected_country)
            & (catalog.topic_id == selected_topic)
        )[0]
        interest_slots = max(1, round(config.route_k * interest_fraction))
        width = min(len(candidates), interest_slots)
        if not width:
            continue
        pool = candidates[torch.topk(score[candidates], width).indices]
        offset = torch.arange(width, device=catalog.item_id.device)[None]
        start = torch.remainder(
            requests.request_id[target] * 503
            + requests.user_id[target] * 1_009,
            width,
        )
        chosen = pool[torch.remainder(start[:, None] + offset, width)]
        item[target, :width] = chosen
        values[target, :width] = score[chosen]
    return item, values
