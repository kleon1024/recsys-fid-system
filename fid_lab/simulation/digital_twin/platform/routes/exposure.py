"""Observable cross-request exposure filters for Feed retrieval."""

from __future__ import annotations

import torch

from ...contracts import PlatformRequestBatch, Surface
from ..projection import PlatformProjectionState


DEDUP_REQUEST_CHUNK = 512


def recently_exposed(
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    route_item: torch.Tensor,
    window_ticks: int,
) -> torch.Tensor:
    duplicate = torch.zeros_like(route_item, dtype=torch.bool)
    for start in range(0, len(requests.user_id), DEDUP_REQUEST_CHUNK):
        stop = min(start + DEDUP_REQUEST_CHUNK, len(requests.user_id))
        user = requests.user_id[start:stop]
        history_item = state.user_feed_exposure_item[user]
        history_time = state.user_feed_exposure_time[user]
        age = requests.event_time[start:stop, None] - history_time
        selected = (
            (history_item >= 0) & (age >= 0) & (age <= window_ticks)
        )
        duplicate[start:stop] = _exact_membership(
            route_item[start:stop], history_item, selected,
        )
    return _feed_only(requests, duplicate)


def exposed_in_current_session(
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    route_item: torch.Tensor,
) -> torch.Tensor:
    duplicate = torch.zeros_like(route_item, dtype=torch.bool)
    for start in range(0, len(requests.user_id), DEDUP_REQUEST_CHUNK):
        stop = min(start + DEDUP_REQUEST_CHUNK, len(requests.user_id))
        user = requests.user_id[start:stop]
        history_item = state.user_feed_exposure_item[user]
        history_time = state.user_feed_exposure_time[user]
        session_start = state.user_session_start_time[user]
        selected = (
            (history_item >= 0)
            & (history_time >= session_start[:, None])
            & (history_time <= requests.event_time[start:stop, None])
        )
        duplicate[start:stop] = _exact_membership(
            route_item[start:stop], history_item, selected,
        )
    return _feed_only(requests, duplicate)


def _exact_membership(
    route_item: torch.Tensor,
    history_item: torch.Tensor,
    selected_history: torch.Tensor,
) -> torch.Tensor:
    rows = len(route_item)
    if route_item.numel() == 0:
        return torch.zeros_like(route_item, dtype=torch.bool)
    flattened = route_item.reshape(rows, -1)
    maximum = torch.maximum(
        flattened.clamp_min(0).max(), history_item.clamp_min(0).max(),
    )
    item_base = maximum + 1
    row = torch.arange(rows, device=route_item.device, dtype=torch.long)
    history_key = (
        row[:, None] * item_base + history_item.clamp_min(0)
    )[selected_history]
    if not len(history_key):
        return torch.zeros_like(route_item, dtype=torch.bool)
    history_key = torch.sort(history_key).values
    candidate_valid = flattened >= 0
    candidate_key = row[:, None] * item_base + flattened.clamp_min(0)
    location = torch.searchsorted(history_key, candidate_key)
    location = location.clamp_max(len(history_key) - 1)
    duplicate = candidate_valid & (history_key[location] == candidate_key)
    return duplicate.reshape_as(route_item)


def _feed_only(
    requests: PlatformRequestBatch,
    duplicate: torch.Tensor,
) -> torch.Tensor:
    feed = requests.surface == int(Surface.FEED)
    return duplicate & feed.reshape((-1,) + (1,) * (duplicate.ndim - 1))
