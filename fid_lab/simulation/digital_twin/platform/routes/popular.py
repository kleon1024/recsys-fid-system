"""Observable country and recent-interest Popular retrieval routes."""

from __future__ import annotations

import torch

from ...catalog import PublicCatalog
from ...contracts import EventType, PlatformRequestBatch, Surface
from ..indexes.contracts import RetrievalConfig
from ..projection import PlatformProjectionState
from ..sequences import resolve_user_sequence
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


def _recent_interest_topics(
    catalog: PublicCatalog,
    config: RetrievalConfig,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = resolve_user_sequence(
        state, requests.user_id, requests.event_time,
    )
    history_item = sequence.item_id
    event_type = sequence.event_type
    event_time = sequence.event_time
    strength = torch.zeros_like(event_time, dtype=torch.float)
    for candidate, value in (
        (EventType.PLAY_3S, 1.0),
        (EventType.LONG_VIEW, 2.0),
        (EventType.COMPLETE, 2.5),
        (EventType.CLICK, 2.0),
        (EventType.LIKE, 3.0),
        (EventType.COMMENT, 3.5),
        (EventType.FAVORITE, 3.5),
        (EventType.SHARE, 4.0),
        (EventType.FOLLOW, 4.0),
    ):
        strength = torch.where(
            event_type == int(candidate), torch.full_like(strength, value),
            strength,
        )
    positive = strength > 0.0
    valid = sequence.valid & positive
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
    width = min(3, topic_score.shape[1])
    values, selected_topic = torch.topk(topic_score, width, dim=1)
    selected_topic = torch.where(
        values > 0.0, selected_topic, torch.full_like(selected_topic, -1),
    )
    return selected_topic, values


def interest_popular_candidates(
    catalog: PublicCatalog,
    config: RetrievalConfig,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    score: torch.Tensor,
    *,
    interest_fraction: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve quality-preserving Popular pools across top strong interests."""
    item = torch.full(
        (len(requests.user_id), config.route_k),
        -1,
        device=catalog.item_id.device,
        dtype=torch.long,
    )
    values = torch.full_like(item, -torch.inf, dtype=torch.float)
    feed = requests.surface == int(Surface.FEED)
    country = state.user_country[requests.user_id]
    topics, topic_values = _recent_interest_topics(
        catalog, config, requests, state,
    )
    eligible_base = _eligible_feed_items(catalog, state)
    interest_slots = max(1, round(config.route_k * interest_fraction))
    topics_per_user = topics.shape[1]
    slot_width = max(1, interest_slots // topics_per_user)
    topic_count = int(catalog.topic_id.max()) + 1
    for rank in range(topics_per_user):
        topic = topics[:, rank]
        segment = country * topic_count + topic
        available = feed & (country >= 0) & (topic >= 0)
        for key in torch.unique(segment[available]).tolist():
            target = torch.where(available & (segment == key))[0]
            selected_country = int(country[target[0]])
            selected_topic = int(topic[target[0]])
            candidates = torch.where(
                eligible_base
                & (state.item_country == selected_country)
                & (catalog.topic_id == selected_topic)
            )[0]
            pool_width = min(
                len(candidates), slot_width * config.popular_pool_multiplier,
            )
            width = min(pool_width, slot_width)
            if not width:
                continue
            pool = candidates[
                torch.topk(score[candidates], pool_width).indices
            ]
            offset = torch.arange(width, device=catalog.item_id.device)[None]
            start = torch.remainder(
                requests.request_id[target] * 503
                + requests.user_id[target] * 1_009
                + rank * 257,
                pool_width,
            )
            chosen = pool[torch.remainder(start[:, None] + offset, pool_width)]
            begin = rank * slot_width
            end = min(begin + width, interest_slots)
            chosen = chosen[:, : end - begin]
            item[target, begin:end] = chosen
            values[target, begin:end] = (
                score[chosen] + 0.001 * topic_values[target, rank, None]
            )
    return item, values
