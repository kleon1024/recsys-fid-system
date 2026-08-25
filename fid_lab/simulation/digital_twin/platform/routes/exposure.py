"""Observable cross-request exposure filters for Feed retrieval."""

from __future__ import annotations

import torch

from ...contracts import PlatformRequestBatch, Surface
from ..state.exposure_bloom import ExposureBloomConfig, contains_exposure
from ..projection import PlatformProjectionState


DEDUP_REQUEST_CHUNK = 2_048


def recently_exposed(
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    route_item: torch.Tensor,
    window_ticks: int,
) -> torch.Tensor:
    segments = state.user_feed_exposure_bloom.shape[1]
    config = ExposureBloomConfig(
        segments=segments,
        bits_per_segment=state.user_feed_exposure_bloom.shape[2] * 8,
        segment_ticks=int(state.feed_exposure_bloom_segment_ticks),
    )
    duplicate = contains_exposure(
        state.user_feed_exposure_bloom,
        state.feed_exposure_bloom_epoch,
        requests.user_id,
        route_item,
        requests.event_time,
        config,
        window_ticks=window_ticks,
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
    if not selected_history.any():
        return torch.zeros_like(route_item, dtype=torch.bool)
    sentinel = torch.maximum(
        flattened.clamp_min(0).max(), history_item.clamp_min(0).max(),
    ) + 1
    searchable = torch.where(
        selected_history, history_item, sentinel,
    ).sort(dim=1).values
    candidate_valid = flattened >= 0
    candidate = flattened.clamp_min(0).contiguous()
    location = torch.searchsorted(searchable, candidate)
    location = location.clamp_max(searchable.shape[1] - 1)
    duplicate = candidate_valid & (
        torch.gather(searchable, 1, location) == candidate
    )
    return duplicate.reshape_as(route_item)


def _feed_only(
    requests: PlatformRequestBatch,
    duplicate: torch.Tensor,
) -> torch.Tensor:
    feed = requests.surface == int(Surface.FEED)
    return duplicate & feed.reshape((-1,) + (1,) * (duplicate.ndim - 1))
