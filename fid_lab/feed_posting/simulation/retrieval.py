"""Feed-posting candidate routes and request-level RRF merge."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts import FEED_POSTING_ROUTES
from .world import category_prompt, hidden_utility


@dataclass(frozen=True)
class FeedPostingCandidates:
    prompt_ids: torch.Tensor
    route_bits: torch.Tensor
    recall_scores: torch.Tensor
    audit_oracle_prompt: torch.Tensor
    audit_oracle_recalled: torch.Tensor


def _route_items(world):
    config, requests = world.config, world.requests
    rank = torch.arange(config.route_candidates, device=requests.request_id.device)
    request = requests.request_id[:, None]
    trending_category = torch.remainder(rank[None, :] * 5 + 2, config.categories)
    trending = category_prompt(
        config, trending_category, rank[None, :]
    ).expand(config.requests, -1)
    i2i = category_prompt(
        config, requests.recent_category[:, None],
        request * 17 + rank[None, :] * 11,
    )
    creator = category_prompt(
        config, requests.creator_category[:, None],
        request * 19 + rank[None, :] * 13,
    )
    observed_category = (
        requests.observed_profile @ world.category_basis.T
    ).argmax(1)
    semantic = category_prompt(
        config, observed_category[:, None],
        request * 23 + rank[None, :] * 17,
    )
    return torch.stack((trending, i2i, creator, semantic), dim=1)


def _deduplicated_top(route_items, route_score, width):
    flat_items, flat_scores = route_items.flatten(1), route_score.flatten(1)
    order = torch.argsort(flat_scores, dim=1, descending=True, stable=True)
    sorted_items = flat_items.gather(1, order)
    sorted_scores = flat_scores.gather(1, order)
    duplicate = torch.zeros_like(sorted_items, dtype=torch.bool)
    for position in range(1, sorted_items.shape[1]):
        duplicate[:, position] = (
            sorted_items[:, :position] == sorted_items[:, position : position + 1]
        ).any(dim=1)
    sorted_scores = sorted_scores.masked_fill(duplicate, -1e9)
    positions = torch.topk(sorted_scores, width, dim=1).indices
    return sorted_items.gather(1, positions), sorted_scores.gather(1, positions)


def _audit_oracle(world):
    audit_rank = torch.arange(32, device=world.requests.request_id.device)[None, :]
    latent_category = (
        world.requests.latent_intent @ world.category_basis.T
    ).argmax(1)
    audit = category_prompt(
        world.config, latent_category[:, None],
        world.requests.request_id[:, None] * 29 + audit_rank * 19,
    )
    score = hidden_utility(world, audit)
    return audit.gather(1, score.argmax(1, keepdim=True)).squeeze(1)


def retrieve(world, enabled_routes):
    unknown = set(enabled_routes) - set(FEED_POSTING_ROUTES)
    if unknown:
        raise ValueError(f"unsupported Feed-posting routes: {sorted(unknown)}")
    route_items = _route_items(world)
    ranks = torch.arange(
        1, world.config.route_candidates + 1, device=route_items.device
    ).float()
    weights = torch.tensor([0.90, 0.95, 1.00, 1.00], device=route_items.device)
    score = weights[None, :, None] / (20.0 + ranks[None, None, :])
    valid = torch.tensor(
        [name in enabled_routes for name in FEED_POSTING_ROUTES],
        device=route_items.device,
    )[None, :, None]
    score = score.expand(len(route_items), -1, -1).masked_fill(~valid, -1e9)
    prompt_ids, recall_scores = _deduplicated_top(
        route_items, score, world.config.merged_candidates
    )
    route_bits = torch.zeros_like(prompt_ids)
    for route_index in range(len(FEED_POSTING_ROUTES)):
        hit = (
            prompt_ids[:, :, None] == route_items[:, route_index, None, :]
        ).any(dim=2)
        route_bits |= hit.long() << route_index
    oracle = _audit_oracle(world)
    return FeedPostingCandidates(
        prompt_ids, route_bits, recall_scores, oracle,
        (prompt_ids == oracle[:, None]).any(1),
    )
