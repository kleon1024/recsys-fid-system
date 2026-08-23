"""Device-resident multi-route recall, RRF merge, and coarse truncation."""

from __future__ import annotations

import torch


ROUTE_NAMES = (
    "ann",
    "graph",
    "geo",
    "fresh",
    "long_tail",
    "popular",
    "post_search",
    "retarget",
)
ROUTE_WEIGHTS = (1.00, 0.85, 0.75, 0.65, 0.60, 0.55, 0.95, 0.90)


def _hashed_items(user_ids, step, count, salt, catalog_size, seed):
    positions = torch.arange(count, device=user_ids.device)[None, :]
    values = (
        user_ids[:, None] * 1_103_515_245
        + positions * 12_345
        + step * 48_271
        + salt * 7_919
        + seed * 503
    )
    return torch.remainder(values, catalog_size).long()


def _topic_aligned(items, topic, topics, catalog_size):
    blocks = max(catalog_size // topics, 1)
    return torch.remainder(items // topics, blocks) * topics + topic[:, None]


def _sampled_top(items, score, limit):
    positions = torch.topk(score, limit, dim=1).indices
    return items.gather(1, positions)


def _route_candidates(config, state, catalog, step):
    users = len(state["active"])
    route_k = config.route_candidates
    pool_k = route_k * config.route_oversample
    user_ids = state["user_ids"]
    target_topic = state["observed_interest"].argmax(dim=1)
    graph_topic = torch.where(
        state["last_topic"] >= 0, state["last_topic"], target_topic
    )
    routes = []

    ann_pool = _hashed_items(user_ids, step, pool_k, 1, catalog.size, config.seed)
    ann_topics = catalog.topics[ann_pool]
    ann_score = torch.einsum("bkd,bd->bk", ann_topics, state["observed_interest"])
    routes.append(_sampled_top(ann_pool, ann_score, route_k))

    graph_seed = _hashed_items(user_ids, step, route_k, 2, catalog.size, config.seed)
    routes.append(
        _topic_aligned(graph_seed, graph_topic, config.topics, catalog.size)
    )

    geo_pool = _hashed_items(user_ids, step, pool_k, 3, catalog.size, config.seed)
    geo_affinity = torch.einsum(
        "bkd,bd->bk", catalog.topics[geo_pool], state["observed_interest"]
    )
    geo_score = (
        2.0 * (catalog.city[geo_pool] == state["city"][:, None]).float()
        + 0.55 * catalog.quality[geo_pool]
        + 0.35 * geo_affinity
    )
    routes.append(_sampled_top(geo_pool, geo_score, route_k))

    fresh_pool = _hashed_items(user_ids, step, pool_k, 4, catalog.size, config.seed)
    routes.append(
        _sampled_top(fresh_pool, catalog.freshness[fresh_pool], route_k)
    )

    tail_pool = _hashed_items(user_ids, step, pool_k, 5, catalog.size, config.seed)
    tail_affinity = torch.einsum(
        "bkd,bd->bk", catalog.topics[tail_pool], state["observed_interest"]
    )
    tail_score = (
        0.45 * catalog.quality[tail_pool]
        + 0.35 * (1.0 - catalog.popularity[tail_pool])
        + 0.20 * tail_affinity
    )
    routes.append(_sampled_top(tail_pool, tail_score, route_k))

    popular = torch.topk(catalog.popularity, route_k).indices
    routes.append(popular[None, :].expand(users, route_k))

    search_seed = _hashed_items(user_ids, step, route_k, 7, catalog.size, config.seed)
    search = _topic_aligned(
        search_seed, state["search_topic"], config.topics, catalog.size
    )
    routes.append(search)

    retarget_seed = _hashed_items(user_ids, step, route_k, 8, catalog.size, config.seed)
    retarget = _topic_aligned(
        retarget_seed, graph_topic, config.topics, catalog.size
    )
    retarget[:, 0] = state["retarget_item"].clamp_min(0)
    routes.append(retarget)

    items = torch.stack(routes, dim=1)
    valid = torch.ones_like(items, dtype=torch.bool)
    valid[:, 6, :] &= state["search_ttl"][:, None] > 0
    valid[:, 7, :] &= state["retarget_item"][:, None] >= 0
    for position in range(1, route_k):
        duplicate = (
            items[:, :, :position] == items[:, :, position : position + 1]
        ).any(dim=2)
        valid[:, :, position] &= ~duplicate
    return items, valid


def _rrf_merge(config, route_items, route_valid):
    users, routes, route_k = route_items.shape
    rank = torch.arange(1, route_k + 1, device=route_items.device).float()
    weights = torch.tensor(
        ROUTE_WEIGHTS, device=route_items.device, dtype=torch.float32
    )
    scores = weights[None, :, None] / (20.0 + rank[None, None, :])
    scores = scores.expand(users, routes, route_k) * route_valid.float()
    bits = (
        2 ** torch.arange(routes, device=route_items.device, dtype=torch.long)
    )[None, :, None].expand(users, routes, route_k)
    bits = bits * route_valid.long()

    flat_items = route_items.reshape(users, -1)
    flat_scores = scores.reshape(users, -1)
    flat_bits = bits.reshape(users, -1)
    sorted_items, order = torch.sort(flat_items, dim=1)
    sorted_scores = flat_scores.gather(1, order)
    sorted_bits = flat_bits.gather(1, order)
    starts = torch.ones_like(sorted_items, dtype=torch.bool)
    starts[:, 1:] = sorted_items[:, 1:] != sorted_items[:, :-1]
    groups = starts.long().cumsum(dim=1) - 1
    width = flat_items.shape[1]
    merged_scores = torch.zeros(users, width, device=route_items.device)
    merged_bits = torch.zeros(
        users, width, device=route_items.device, dtype=torch.long
    )
    merged_items = torch.zeros_like(merged_bits)
    merged_scores.scatter_add_(1, groups, sorted_scores)
    merged_bits.scatter_add_(1, groups, sorted_bits)
    merged_items.scatter_(1, groups, sorted_items)
    valid_counts = torch.zeros_like(merged_bits)
    valid_counts.scatter_add_(1, groups, (sorted_scores > 0).long())
    valid_groups = valid_counts > 0
    merged_scores.masked_fill_(~valid_groups, -1e9)
    keep = min(config.merged_candidates, width)
    positions = torch.topk(merged_scores, keep, dim=1).indices
    return (
        merged_items.gather(1, positions),
        merged_scores.gather(1, positions),
        merged_bits.gather(1, positions),
        valid_groups.sum(dim=1),
        valid_groups.gather(1, positions),
    )


def _audit_oracle(config, state, catalog, step, route_items, route_valid):
    random_items = _hashed_items(
        state["user_ids"], step, config.audit_candidates, 19, catalog.size,
        config.seed,
    )
    route_items = route_items.flatten(1)
    route_valid = route_valid.flatten(1)
    items = torch.cat((route_items, random_items), dim=1)
    affinity = torch.einsum(
        "bkd,bd->bk", catalog.topics[items], state["interest"]
    )
    utility = affinity + 0.45 * catalog.quality[items]
    utility[:, : route_items.shape[1]].masked_fill_(~route_valid, -1e9)
    position = utility.argmax(dim=1)
    batch = torch.arange(len(items), device=items.device)
    return items[batch, position], utility[batch, position]


def build_candidate_graph(config, state, catalog, step):
    """Return the coarse-surviving candidate set plus stage lineage."""
    route_items, route_valid = _route_candidates(config, state, catalog, step)
    merged, rrf_score, route_bits, unique_count, merged_valid = _rrf_merge(
        config, route_items, route_valid
    )
    observed_affinity = torch.einsum(
        "bkd,bd->bk", catalog.topics[merged], state["observed_interest"]
    )
    same_city = (catalog.city[merged] == state["city"][:, None]).float()
    coarse_score = (
        0.46 * observed_affinity
        + 0.18 * catalog.quality[merged]
        + 0.12 * catalog.popularity[merged]
        + 0.10 * catalog.freshness[merged]
        + 0.08 * same_city
        + 0.04 * rrf_score
        + 0.02 * catalog.poi_quality[merged]
    )
    coarse_score.masked_fill_(~merged_valid, -1e9)
    coarse_positions = torch.topk(coarse_score, config.candidates, dim=1).indices
    coarse_items = merged.gather(1, coarse_positions)
    audit_item, audit_utility = _audit_oracle(
        config, state, catalog, step, route_items, route_valid
    )
    audit_in_recall = (merged == audit_item[:, None]).any(dim=1)
    audit_in_coarse = (coarse_items == audit_item[:, None]).any(dim=1)
    true_utility = (
        torch.einsum("bkd,bd->bk", catalog.topics[merged], state["interest"])
        + 0.45 * catalog.quality[merged]
    )
    return {
        "item_ids": coarse_items,
        "route_bits": route_bits.gather(1, coarse_positions),
        "recall_score": rrf_score.gather(1, coarse_positions),
        "coarse_score": coarse_score.gather(1, coarse_positions),
        "route_valid_counts": route_valid.sum(dim=2),
        "unique_recall_count": unique_count,
        "coarse_pass_fraction": torch.full(
            (len(coarse_items),),
            config.candidates / config.merged_candidates,
            device=coarse_items.device,
        ),
        "audit_oracle_item": audit_item,
        "audit_oracle_utility": audit_utility,
        "audit_oracle_in_recall": audit_in_recall,
        "audit_oracle_in_coarse": audit_in_coarse,
        "merged_oracle_utility": true_utility.max(dim=1).values,
        "recalled_item_ids": merged,
        "recalled_scores": rrf_score,
        "recalled_route_bits": route_bits,
        "recalled_coarse_scores": coarse_score,
    }
