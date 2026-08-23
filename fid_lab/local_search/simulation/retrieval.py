"""Local Search routes, RRF merge, and audit-oracle coverage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts import LOCAL_SEARCH_ROUTES
from .world import category_poi, hidden_utility


@dataclass(frozen=True)
class SearchCandidates:
    poi_ids: torch.Tensor
    route_bits: torch.Tensor
    recall_scores: torch.Tensor
    audit_oracle_poi: torch.Tensor
    audit_oracle_recalled: torch.Tensor


def query_tower_features(world):
    requests = world.requests
    numeric = torch.stack((
        requests.price_preference, requests.urgency, requests.activity,
        (requests.source == 0).float(), (requests.source == 1).float(),
        (requests.source == 2).float(),
    ), dim=1)
    return torch.cat((
        requests.observed_query, requests.history_summary, numeric
    ), dim=1)


def item_tower_features(world, poi_ids):
    catalog = world.catalog
    numeric = torch.stack((
        catalog.quality[poi_ids], catalog.popularity[poi_ids],
        catalog.price[poi_ids], catalog.availability[poi_ids],
        catalog.risk[poi_ids],
    ), dim=-1)
    return torch.cat((catalog.semantic[poi_ids], numeric), dim=-1)


def _route_items(world, semantic_tower=None):
    config, requests = world.config, world.requests
    rank = torch.arange(config.route_candidates, device=requests.request_id.device)
    request = requests.request_id[:, None]
    lexical = category_poi(
        config, requests.observed_category[:, None], request * 17 + rank * 13
    )
    geo_pool = torch.remainder(
        request * 19
        + torch.arange(64, device=request.device)[None, :] * 43,
        config.pois,
    )
    geo_distance = (
        (world.catalog.latitude[geo_pool] - requests.latitude[:, None]).square()
        + (world.catalog.longitude[geo_pool] - requests.longitude[:, None]).square()
    )
    geo = geo_pool.gather(
        1, torch.topk(-geo_distance, config.route_candidates, dim=1).indices
    )
    history = category_poi(
        config, requests.history_category[:, None], request * 29 + rank * 17
    )
    retarget = torch.remainder(
        requests.prior_detail_poi[:, None] + rank[None, :] * config.categories,
        config.pois,
    )
    semantic_pool = category_poi(
        config, requests.observed_category[:, None],
        request * 31 + torch.arange(64, device=request.device)[None, :] * 19,
    )
    if semantic_tower is None:
        semantic_score = torch.einsum(
            "bpd,bd->bp", world.catalog.semantic[semantic_pool],
            requests.observed_query,
        )
    else:
        query = semantic_tower.encode_query(query_tower_features(world))
        item = semantic_tower.encode_item(item_tower_features(world, semantic_pool))
        semantic_score = torch.einsum("bpd,bd->bp", item, query)
    semantic_position = torch.topk(
        semantic_score, config.route_candidates, dim=1
    ).indices
    semantic = semantic_pool.gather(1, semantic_position)
    return torch.stack((lexical, geo, semantic, history, retarget), dim=1)


def _deduplicated_top(items, scores, width):
    flat_items, flat_scores = items.flatten(1), scores.flatten(1)
    order = torch.argsort(flat_scores, dim=1, descending=True, stable=True)
    sorted_items = flat_items.gather(1, order)
    sorted_scores = flat_scores.gather(1, order)
    duplicate = torch.zeros_like(sorted_items, dtype=torch.bool)
    for position in range(1, sorted_items.shape[1]):
        duplicate[:, position] = (
            sorted_items[:, :position] == sorted_items[:, position : position + 1]
        ).any(1)
    sorted_scores = sorted_scores.masked_fill(duplicate, -1e9)
    top = torch.topk(sorted_scores, width, dim=1).indices
    return sorted_items.gather(1, top), sorted_scores.gather(1, top)


def _audit_oracle(world):
    rank = torch.arange(64, device=world.requests.request_id.device)[None, :]
    audit = category_poi(
        world.config, world.requests.latent_category[:, None],
        world.requests.request_id[:, None] * 37 + rank * 23,
    )
    score = hidden_utility(world, audit)
    return audit.gather(1, score.argmax(1, keepdim=True)).squeeze(1)


def retrieve(world, enabled_routes, semantic_tower=None):
    unknown = set(enabled_routes) - set(LOCAL_SEARCH_ROUTES)
    if unknown:
        raise ValueError(f"unsupported Local Search routes: {sorted(unknown)}")
    items = _route_items(world, semantic_tower)
    ranks = torch.arange(
        1, world.config.route_candidates + 1, device=items.device
    ).float()
    route_weights = torch.tensor(
        [1.00, 0.95, 1.05, 1.00, 0.90], device=items.device
    )
    scores = route_weights[None, :, None] / (20.0 + ranks[None, None, :])
    valid = torch.tensor(
        [name in enabled_routes for name in LOCAL_SEARCH_ROUTES],
        device=items.device,
    )[None, :, None]
    if "retarget" in enabled_routes:
        valid = valid.expand(len(items), -1, -1).clone()
        valid[:, 4, :] &= world.requests.retarget_eligible[:, None]
    scores = scores.expand(len(items), -1, -1).masked_fill(~valid, -1e9)
    fallback_rank = torch.arange(
        2 * world.config.merged_candidates, device=items.device
    )[None, :]
    fallback = torch.remainder(
        world.requests.request_id[:, None] * 53 + fallback_rank * 97,
        world.config.pois,
    )
    fallback_score = -0.01 - fallback_rank.float() * 1e-6
    merge_items = torch.cat((items.flatten(1), fallback), dim=1)
    merge_scores = torch.cat((scores.flatten(1), fallback_score.expand(
        len(items), -1
    )), dim=1)
    poi_ids, recall_scores = _deduplicated_top(
        merge_items, merge_scores, world.config.merged_candidates
    )
    route_bits = torch.zeros_like(poi_ids)
    for index in range(len(LOCAL_SEARCH_ROUTES)):
        hit = (poi_ids[:, :, None] == items[:, index, None, :]).any(2)
        route_bits |= hit.long() << index
    oracle = _audit_oracle(world)
    return SearchCandidates(
        poi_ids, route_bits, recall_scores, oracle,
        (poi_ids == oracle[:, None]).any(1),
    )
