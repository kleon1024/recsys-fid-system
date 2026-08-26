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
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
) -> torch.Tensor:
    user_id = requests.user_id
    history_item = state.user_history_item[user_id]
    event_type = state.user_history_event_type[user_id]
    event_time = state.user_history_event_time[user_id]
    positive = torch.zeros_like(event_type, dtype=torch.bool)
    for candidate in (
        EventType.PLAY_3S,
        EventType.LONG_VIEW,
        EventType.COMPLETE,
        EventType.CLICK,
        EventType.LIKE,
        EventType.FAVORITE,
        EventType.SHARE,
    ):
        positive |= event_type == int(candidate)
    valid = positive & (history_item >= 0)
    selected_time = torch.where(
        valid, event_time, torch.full_like(event_time, -1),
    )
    slot = selected_time.argmax(dim=1)
    selected_item = torch.gather(history_item, 1, slot[:, None]).squeeze(1)
    selected_item = torch.where(
        valid.any(dim=1), selected_item, torch.full_like(selected_item, -1),
    )
    return torch.where(
        selected_item >= 0,
        catalog.topic_id[selected_item.clamp_min(0)],
        torch.full_like(selected_item, -1),
    )


def interest_popular_candidates(
    catalog: PublicCatalog,
    config: RetrievalConfig,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve popular items inside the latest observable interest segment."""
    item, values = popular_candidates(
        catalog, config, requests, state, score,
    )
    feed = requests.surface == int(Surface.FEED)
    country = state.user_country[requests.user_id]
    topic = _recent_interest_topic(catalog, requests, state)
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
        width = min(len(candidates), config.route_k)
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
