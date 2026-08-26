"""Observable event closure shared by every sample materializer."""

from __future__ import annotations

import torch

from ..contracts import AppEventBatch, EventType, Surface


PUBLISH_QUEUE_OUTCOME_TYPES = (
    EventType.SURFACE_ENTRY,
    EventType.CREATE,
    EventType.PUBLISH,
)


def select_joiner_events(
    events: AppEventBatch,
    *,
    request_id: torch.Tensor,
    user_id: torch.Tensor,
    request_time: torch.Tensor,
    publish_window_ticks: int,
) -> AppEventBatch:
    """Keep same-request actions and later posting outcomes for Feed queues."""
    if publish_window_ticks <= 0:
        raise ValueError("publish event window must be positive")
    if not len(events.request_id) or not len(request_id):
        return AppEventBatch.empty(events.request_id.device)
    same_request = torch.isin(events.request_id, request_id)
    outcome_types = torch.tensor(
        [int(event_type) for event_type in PUBLISH_QUEUE_OUTCOME_TYPES],
        device=events.event_type.device,
    )
    cross_request_publish = (
        torch.isin(events.user_id, user_id)
        & (events.surface == int(Surface.POSTING))
        & torch.isin(events.event_type, outcome_types)
        & (events.event_time >= int(request_time.min()))
        & (events.event_time <= int(request_time.max()) + publish_window_ticks)
    )
    return events.select(same_request | cross_request_publish)
