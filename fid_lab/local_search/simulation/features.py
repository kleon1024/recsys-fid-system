"""Point-in-time observable query-POI features and rule baseline."""

from __future__ import annotations

import torch


FEATURE_NAMES = (
    "query_semantic", "history_semantic", "lexical_category", "city_match",
    "distance", "price_match", "quality", "availability", "popularity",
    "risk", "urgency", "activity", "source_typed", "source_feed",
    "source_map", "route_semantic", "route_history", "route_retarget",
)


def candidate_features(world, candidates):
    catalog, requests = world.catalog, world.requests
    poi_ids = candidates.poi_ids
    query = torch.einsum(
        "bkd,bd->bk", catalog.semantic[poi_ids], requests.observed_query
    )
    history = torch.einsum(
        "bkd,bd->bk", catalog.semantic[poi_ids], requests.history_summary
    )
    distance = torch.sqrt(
        (catalog.latitude[poi_ids] - requests.latitude[:, None]).square()
        + (catalog.longitude[poi_ids] - requests.longitude[:, None]).square()
    )
    price_match = 1.0 - (
        catalog.price[poi_ids] - requests.price_preference[:, None]
    ).abs()
    return torch.stack((
        query,
        history,
        (catalog.category[poi_ids] == requests.observed_category[:, None]).float(),
        (catalog.city[poi_ids] == requests.city[:, None]).float(),
        distance,
        price_match,
        catalog.quality[poi_ids],
        catalog.availability[poi_ids],
        catalog.popularity[poi_ids],
        catalog.risk[poi_ids],
        requests.urgency[:, None].expand_as(query),
        requests.activity[:, None].expand_as(query),
        (requests.source[:, None] == 0).float().expand_as(query),
        (requests.source[:, None] == 1).float().expand_as(query),
        (requests.source[:, None] == 2).float().expand_as(query),
        ((candidates.route_bits & (1 << 2)) > 0).float(),
        ((candidates.route_bits & (1 << 3)) > 0).float(),
        ((candidates.route_bits & (1 << 4)) > 0).float(),
    ), dim=2)


def rule_score(features):
    return (
        0.42 * features[:, :, 0]
        + 0.22 * features[:, :, 2]
        + 0.18 * features[:, :, 3]
        - 0.32 * features[:, :, 4]
        + 0.16 * features[:, :, 6]
        + 0.12 * features[:, :, 7]
        - 0.12 * features[:, :, 9]
    )
