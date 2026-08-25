"""Observable cross-request exposure filters for Feed retrieval."""

from __future__ import annotations

import torch

from ...contracts import PlatformRequestBatch, Surface
from ..projection import PlatformProjectionState


def recently_exposed(
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    route_item: torch.Tensor,
    window_ticks: int,
) -> torch.Tensor:
    history_item = state.user_exposure_item[requests.user_id]
    history_time = state.user_exposure_time[requests.user_id]
    age = requests.event_time[:, None] - history_time
    recent = (history_item >= 0) & (age >= 0) & (age <= window_ticks)
    return _route_matches_history(requests, route_item, history_item, recent)


def exposed_in_current_session(
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    route_item: torch.Tensor,
) -> torch.Tensor:
    history_item = state.user_exposure_item[requests.user_id]
    history_time = state.user_exposure_time[requests.user_id]
    session_start = state.user_session_start_time[requests.user_id]
    current_session = (
        (history_item >= 0)
        & (history_time >= session_start[:, None])
        & (history_time <= requests.event_time[:, None])
    )
    return _route_matches_history(
        requests, route_item, history_item, current_session,
    )


def _route_matches_history(
    requests: PlatformRequestBatch,
    route_item: torch.Tensor,
    history_item: torch.Tensor,
    selected_history: torch.Tensor,
) -> torch.Tensor:
    flattened = route_item.reshape(len(route_item), -1)
    duplicate = (
        (flattened[:, :, None] == history_item[:, None, :])
        & selected_history[:, None, :]
    ).any(dim=2).reshape_as(route_item)
    feed = requests.surface == int(Surface.FEED)
    return duplicate & feed[:, None, None]
