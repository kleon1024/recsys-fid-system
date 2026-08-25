"""Search query, reformulation, success and post-search Feed audit."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...contracts import AppEventBatch, EventType, Surface


@dataclass(frozen=True)
class SearchSessionAudit:
    queries: int
    successes: int
    reformulations: int
    abandonments: int
    post_search_feed_entries: int


def audit_search_sessions(events: AppEventBatch) -> SearchSessionAudit:
    query = events.event(EventType.QUERY)
    success = events.event(EventType.SEARCH_SUCCESS)
    reformulate = events.event(EventType.SEARCH_REFORMULATE)
    abandon = events.event(EventType.SEARCH_ABANDON)
    post_feed = (
        events.event(EventType.SURFACE_ENTRY)
        & (events.surface == int(Surface.FEED))
        & (events.query_id >= 0)
    )
    query_request = events.request_id[query]
    if not torch.isin(events.request_id[success], query_request).all():
        raise ValueError("Search success exists outside a query request")
    if not torch.isin(events.request_id[reformulate], query_request).all():
        raise ValueError("Search reformulation exists outside a query request")
    if not torch.isin(events.request_id[abandon], query_request).all():
        raise ValueError("Search abandonment exists outside a query request")
    if torch.isin(
        events.request_id[reformulate], events.request_id[success],
    ).any():
        raise ValueError("one Search request cannot both succeed and reformulate")
    if torch.isin(
        events.request_id[abandon],
        torch.cat((events.request_id[success], events.request_id[reformulate])),
    ).any():
        raise ValueError("one Search request has conflicting terminal outcomes")
    if not torch.isin(events.query_id[post_feed], events.query_id[query]).all():
        raise ValueError("post-search Feed entry has unknown query lineage")
    return SearchSessionAudit(
        queries=int(query.sum()),
        successes=int(success.sum()),
        reformulations=int(reformulate.sum()),
        abandonments=int(abandon.sum()),
        post_search_feed_entries=int(post_feed.sum()),
    )
