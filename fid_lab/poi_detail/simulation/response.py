"""Quota mixing and module-specific detail-page behavior cascade."""

from __future__ import annotations

import torch

from ..contracts import DETAIL_TASKS
from .world import deterministic_gumbel, deterministic_uniform, hidden_utility


def _quota_exposure(config, scores):
    width = config.candidates_per_module
    related = torch.topk(scores[:, :width], config.exposed_related, dim=1).indices
    product = torch.topk(
        scores[:, width : 2 * width], config.exposed_product, dim=1
    ).indices + width
    review = torch.topk(
        scores[:, 2 * width :], config.exposed_review, dim=1
    ).indices + 2 * width
    return torch.stack((
        related[:, 0], review[:, 0], product[:, 0], related[:, 1],
        review[:, 1], product[:, 1], related[:, 2], related[:, 3],
    ), dim=1)


def _selected_catalog_values(world, kinds, ids, field):
    output = torch.empty_like(ids, dtype=torch.float)
    for kind, catalog in enumerate(world.catalogs):
        mask = kinds == kind
        output[mask] = getattr(catalog, field)[ids[mask]]
    return output


def _cascade(world, kinds, ids, utility, clicked):
    request, config = world.requests, world.config
    deep_probability = torch.sigmoid(
        -1.25 + 0.85 * utility + 0.45 * request.activity
    )
    deep = clicked & (
        deterministic_uniform(request.request_id, ids, 402, config.seed)
        < deep_probability
    )
    transaction_probability = torch.sigmoid(
        -3.4 + 0.85 * utility + 0.9 * request.transaction_propensity
        + 0.55 * _selected_catalog_values(world, kinds, ids, "availability")
    )
    transaction = deep & (kinds != 2) & (
        deterministic_uniform(request.request_id, ids, 403, config.seed)
        < transaction_probability
    )
    negative_probability = torch.sigmoid(
        -3.0 + 1.5 * _selected_catalog_values(world, kinds, ids, "risk")
        + 0.8 * _selected_catalog_values(world, kinds, ids, "toxicity")
    )
    negative = clicked & (
        deterministic_uniform(request.request_id, ids, 404, config.seed)
        < negative_probability
    )
    return deep, transaction, negative


def _labels(world, candidates, top, selected_rank, states):
    clicked, deep, transaction, negative = states
    config = world.config
    labels = torch.zeros(
        config.requests, config.candidates, len(DETAIL_TASKS), device=top.device
    )
    masks = torch.zeros_like(labels)
    exposed = torch.zeros(
        config.requests, config.candidates, dtype=torch.bool, device=top.device
    )
    exposed.scatter_(1, top, True)
    masks[:] = exposed[:, :, None]
    review = candidates.module_kind == 2
    masks[:, :, 2] = exposed & ~review
    batch = torch.arange(config.requests, device=top.device)
    selected = top.gather(1, selected_rank[:, None]).squeeze(1)
    for task, state in enumerate((clicked, deep, transaction, negative)):
        labels[batch[state], selected[state], task] = 1.0
    return labels, masks


def simulate_response(world, candidates, scores):
    config, requests = world.config, world.requests
    top = _quota_exposure(config, scores)
    ids = candidates.entity_ids.gather(1, top)
    kinds = candidates.module_kind.gather(1, top)
    utility = hidden_utility(world, kinds, ids)
    position = torch.arange(config.exposed, device=scores.device).float()
    choice = utility - 0.10 * position[None, :] + deterministic_gumbel(
        requests.request_id[:, None], ids, 400, config.seed
    )
    best, selected_rank = choice.max(1)
    selected_ids = ids.gather(1, selected_rank[:, None]).squeeze(1)
    selected_kinds = kinds.gather(1, selected_rank[:, None]).squeeze(1)
    clicked = best > requests.outside_preference
    selected_utility = utility.gather(1, selected_rank[:, None]).squeeze(1)
    deep, transaction, negative = _cascade(
        world, selected_kinds, selected_ids, selected_utility, clicked
    )
    labels, masks = _labels(
        world, candidates, top, selected_rank,
        (clicked, deep, transaction, negative),
    )
    quality = _selected_catalog_values(
        world, selected_kinds, selected_ids, "quality"
    )
    stay = clicked.float() * (2.5 + 3.0 * torch.sigmoid(selected_utility))
    stay += deep.float() * (2.0 + 2.0 * quality)
    active = torch.clamp(
        0.0007 * deep.float() + 0.0020 * transaction.float()
        - 0.0010 * negative.float(), min=-0.001, max=0.004,
    )
    return {
        "top_indices": top, "labels": labels, "label_masks": masks,
        "clicked": clicked, "deep": deep, "transaction": transaction,
        "negative": negative, "selected_kind": selected_kinds,
        "stay_seconds": stay, "active_day": active,
        "selected_risk": _selected_catalog_values(
            world, selected_kinds, selected_ids, "risk"
        ),
    }
