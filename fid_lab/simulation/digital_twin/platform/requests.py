"""Open observable serving requests from delivered navigation events."""

from __future__ import annotations

import torch

from ..contracts import AppEventBatch, EventType, PlatformRequestBatch


def open_platform_requests(events: AppEventBatch) -> PlatformRequestBatch:
    entry = events.event(EventType.SURFACE_ENTRY)
    request_id = events.request_id[entry]
    query_topic = torch.full_like(request_id, -1)
    query = events.event(EventType.QUERY)
    if query.any() and len(request_id):
        query_request = events.request_id[query]
        if torch.unique(query_request).numel() != len(query_request):
            raise ValueError("a request cannot contain multiple query events")
        ordered, order = torch.sort(query_request)
        location = torch.searchsorted(ordered, request_id).clamp_max(
            len(ordered) - 1
        )
        matched = ordered[location] == request_id
        query_topic[matched] = events.topic_id[query][order][location[matched]]
    return PlatformRequestBatch(
        request_id=request_id,
        user_id=events.user_id[entry],
        surface=events.surface[entry],
        event_time=events.event_time[entry],
        query_topic=query_topic,
        user_creator_id=events.creator_id[entry],
    )
