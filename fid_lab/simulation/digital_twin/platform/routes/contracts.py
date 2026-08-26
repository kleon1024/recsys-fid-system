"""Stable route names and surface eligibility owned outside route algorithms."""

from __future__ import annotations

import torch

from ...contracts import ContentKind, Surface


FEED_ROUTE_NAMES = (
    "random",
    "popular",
    "interest_popular",
    "blended_popular",
    "recent_ann",
    "recent_graph",
    "following",
    "cold_start",
    "hot",
    "evergreen",
)

BUSINESS_ROUTE_NAMES = (
    "local_geo",
    "posting_context",
    "posting_diverse",
    "commerce_intent",
    "live_now",
    "ads_auction",
    "search",
    "search_semantic",
    "retarget",
)

DEFAULT_BUSINESS_ROUTE_NAMES = tuple(
    route for route in BUSINESS_ROUTE_NAMES
    if route not in {"ads_auction", "search_semantic", "posting_diverse"}
)

ROUTE_NAMES = (*FEED_ROUTE_NAMES, *BUSINESS_ROUTE_NAMES)

SURFACE_CONTENT = {
    Surface.FEED: (
        ContentKind.SHORT_VIDEO,
        ContentKind.PHOTO,
        ContentKind.ARTICLE,
        ContentKind.CARD,
        ContentKind.LIVE_ROOM,
        ContentKind.PRODUCT,
        ContentKind.POI,
        ContentKind.AD,
    ),
    Surface.SEARCH: (
        ContentKind.SHORT_VIDEO,
        ContentKind.PHOTO,
        ContentKind.ARTICLE,
        ContentKind.CARD,
        ContentKind.PRODUCT,
        ContentKind.POI,
    ),
    Surface.COMMERCE: (
        ContentKind.PRODUCT,
        ContentKind.LIVE_ROOM,
        ContentKind.AD,
    ),
    Surface.LIVE: (ContentKind.LIVE_ROOM,),
    Surface.LOCAL: (
        ContentKind.SHORT_VIDEO,
        ContentKind.PHOTO,
        ContentKind.CARD,
        ContentKind.PRODUCT,
        ContentKind.POI,
    ),
    Surface.POSTING: (
        ContentKind.POI,
        ContentKind.PRODUCT,
        ContentKind.CREATOR_PROMPT,
    ),
}


def surface_eligibility(
    surface: int | torch.Tensor,
    content_kind: torch.Tensor,
) -> torch.Tensor:
    if isinstance(surface, int):
        allowed = SURFACE_CONTENT[Surface(surface)]
        result = torch.zeros_like(content_kind, dtype=torch.bool)
        for kind in allowed:
            result |= content_kind == int(kind)
        return result
    result = torch.zeros(
        len(surface), content_kind.shape[-1],
        device=content_kind.device,
        dtype=torch.bool,
    )
    for candidate_surface, allowed in SURFACE_CONTENT.items():
        rows = surface == int(candidate_surface)
        if not rows.any():
            continue
        kinds = content_kind[rows]
        for kind in allowed:
            result[rows] |= kinds == int(kind)
    return result
