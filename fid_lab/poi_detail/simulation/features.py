"""Observable module-specific detail-page features and rule score."""

from __future__ import annotations

import torch


FEATURE_NAMES = (
    "intent_semantic", "history_semantic", "category_match", "quality",
    "price_match", "availability", "trust", "informativeness", "toxicity",
    "freshness", "risk", "activity", "transaction_propensity",
    "entry_feed", "entry_search", "entry_map", "kind_related", "kind_product",
    "kind_review",
)


def _catalog_value(world, candidates, field):
    output = torch.empty_like(candidates.entity_ids, dtype=torch.float)
    for kind, catalog in enumerate(world.catalogs):
        mask = candidates.module_kind == kind
        output[mask] = getattr(catalog, field)[candidates.entity_ids[mask]]
    return output


def _semantic(world, candidates):
    shape = (*candidates.entity_ids.shape, world.config.semantic_dim)
    output = torch.empty(shape, device=candidates.entity_ids.device)
    for kind, catalog in enumerate(world.catalogs):
        mask = candidates.module_kind == kind
        output[mask] = catalog.semantic[candidates.entity_ids[mask]]
    return output


def candidate_semantic(world, candidates):
    return _semantic(world, candidates)


def candidate_features(world, candidates):
    requests = world.requests
    semantic = _semantic(world, candidates)
    intent = torch.einsum("bkd,bd->bk", semantic, requests.observed_intent)
    history = torch.einsum("bkd,bd->bk", semantic, requests.history_summary)
    category = torch.empty_like(candidates.entity_ids)
    for kind, catalog in enumerate(world.catalogs):
        mask = candidates.module_kind == kind
        category[mask] = catalog.category[candidates.entity_ids[mask]]
    price = _catalog_value(world, candidates, "price")
    fields = {
        name: _catalog_value(world, candidates, name)
        for name in (
            "quality", "availability", "trust", "informativeness", "toxicity",
            "freshness", "risk",
        )
    }
    return torch.stack((
        intent, history,
        (category == requests.category[:, None]).float(),
        fields["quality"],
        1.0 - (price - requests.price_preference[:, None]).abs(),
        fields["availability"], fields["trust"], fields["informativeness"],
        fields["toxicity"], fields["freshness"], fields["risk"],
        requests.activity[:, None].expand_as(intent),
        requests.transaction_propensity[:, None].expand_as(intent),
        (requests.entry_source[:, None] == 0).float().expand_as(intent),
        (requests.entry_source[:, None] == 1).float().expand_as(intent),
        (requests.entry_source[:, None] == 2).float().expand_as(intent),
        (candidates.module_kind == 0).float(),
        (candidates.module_kind == 1).float(),
        (candidates.module_kind == 2).float(),
    ), dim=2)


def rule_score(features):
    return (
        0.38 * features[:, :, 0] + 0.16 * features[:, :, 2]
        + 0.16 * features[:, :, 3] + 0.10 * features[:, :, 5]
        + 0.10 * features[:, :, 7] - 0.18 * features[:, :, 8]
        - 0.15 * features[:, :, 10]
    )
