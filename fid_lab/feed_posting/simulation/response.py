"""Creator response cascade and downstream Feed value."""

from __future__ import annotations

import torch

from .world import (
    deterministic_gumbel,
    deterministic_uniform,
    hidden_feed_outputs,
    hidden_utility,
)


def _quality_potential(world, prompt):
    return torch.sigmoid(
        0.8 * world.catalog.quality[prompt]
        + 0.7 * world.requests.experience
        + 0.8 * (
            world.catalog.semantic[prompt] * world.requests.latent_intent
        ).sum(1)
        - 0.45 * world.catalog.difficulty[prompt]
    )


def _candidate_quality_labels(world, prompt_ids):
    return torch.sigmoid(
        0.8 * world.catalog.quality[prompt_ids]
        + 0.7 * world.requests.experience[:, None]
        + 0.8 * torch.einsum(
            "bkd,bd->bk", world.catalog.semantic[prompt_ids],
            world.requests.latent_intent,
        )
        - 0.45 * world.catalog.difficulty[prompt_ids]
    )


def _cascade(world, prompt, selected_utility, clicked, selected_hidden=None):
    requests, config = world.requests, world.config
    create_probability = (
        torch.sigmoid(selected_hidden[:, 1])
        if selected_hidden is not None else torch.sigmoid(
            -1.35 + 0.95 * selected_utility + 0.70 * requests.activity
            - 0.55 * requests.fatigue
        )
    )
    created = clicked & (
        deterministic_uniform(requests.request_id, prompt, 203, config.seed)
        < create_probability
    )
    quality = (
        torch.sigmoid(selected_hidden[:, 3])
        if selected_hidden is not None else _quality_potential(world, prompt)
    )
    publish_probability = (
        torch.sigmoid(selected_hidden[:, 2])
        if selected_hidden is not None else torch.sigmoid(
            -0.80 + 1.1 * quality + 0.65 * requests.activity
            - 0.60 * world.catalog.difficulty[prompt]
        )
    )
    published = created & (
        deterministic_uniform(requests.request_id, prompt, 204, config.seed)
        < publish_probability
    )
    return created, published, quality


def _training_labels(world, candidates, top, selected_rank, states, quality):
    clicked, created, published, negative = states
    config = world.config
    labels = torch.zeros(
        len(world.requests.request_id), config.merged_candidates, 5,
        device=top.device,
    )
    masks = torch.ones_like(labels)
    batch = torch.arange(len(world.requests.request_id), device=top.device)
    selected_index = top.gather(1, selected_rank[:, None]).squeeze(1)
    labels[batch[clicked], selected_index[clicked], 0] = 1.0
    labels[batch[created], selected_index[created], 1] = 1.0
    labels[batch[published], selected_index[published], 2] = 1.0
    if world.teacher is None:
        labels[:, :, 3] = (
            _candidate_quality_labels(world, candidates.prompt_ids) > 0.65
        ).float()
    else:
        masks[:, :, 3] = 0.0
        masks[:, :, 4] = 0.0
        labels[batch[published], selected_index[published], 3] = (
            quality[published] > 0.55
        ).float()
        masks[batch[published], selected_index[published], 3] = 1.0
        labels[batch[negative], selected_index[negative], 4] = 1.0
        masks[batch[published], selected_index[published], 4] = 1.0
    return labels, masks


def simulate_response(world, candidates, scores):
    config, requests = world.config, world.requests
    top = torch.topk(scores, config.exposed_candidates, dim=1).indices
    exposed = candidates.prompt_ids.gather(1, top)
    hidden = (
        hidden_feed_outputs(world, exposed)
        if world.teacher is not None else None
    )
    utility = hidden[:, :, 0] if hidden is not None else hidden_utility(world, exposed)
    position = torch.arange(config.exposed_candidates, device=scores.device).float()
    choice = utility - 0.11 * position[None, :] + deterministic_gumbel(
        requests.request_id[:, None], exposed, 201, config.seed
    )
    outside = requests.outside_preference + deterministic_gumbel(
        requests.request_id,
        torch.full_like(requests.request_id, config.prompts + 1),
        202, config.seed,
    )
    best, selected_rank = choice.max(1)
    clicked = best > outside
    prompt = exposed.gather(1, selected_rank[:, None]).squeeze(1)
    selected_utility = utility.gather(1, selected_rank[:, None]).squeeze(1)
    selected_hidden = (
        hidden.gather(
            1, selected_rank[:, None, None].expand(-1, 1, hidden.shape[2])
        ).squeeze(1)
        if hidden is not None else None
    )
    created, published, quality = _cascade(
        world, prompt, selected_utility, clicked, selected_hidden
    )
    risk = (
        torch.sigmoid(selected_hidden[:, 4])
        if selected_hidden is not None else torch.sigmoid(
            -4.2 + 0.9 * world.catalog.saturation[prompt] - 1.5 * quality
        )
    )
    negative = published & (
        deterministic_uniform(requests.request_id, prompt, 205, config.seed)
        < risk
    )
    stay = published.float() * (
        0.70 + 2.0 * quality
        + 0.55 * (1.0 - world.catalog.saturation[prompt])
        + (
            torch.nn.functional.softplus(selected_hidden[:, 5])
            if selected_hidden is not None else 0.0
        )
    )
    active = published.float() * torch.clamp(
        0.0012 * quality + 0.0007 * requests.activity, max=0.003
    )
    labels, label_masks = _training_labels(
        world, candidates, top, selected_rank,
        (clicked, created, published, negative), quality,
    )
    return {
        "top_indices": top,
        "clicked": clicked,
        "created": created,
        "published": published,
        "quality_potential": quality,
        "content_risk": risk,
        "negative": negative,
        "feed_stay_seconds": stay,
        "feed_active_day": active,
        "labels": labels,
        "label_masks": label_masks,
    }
