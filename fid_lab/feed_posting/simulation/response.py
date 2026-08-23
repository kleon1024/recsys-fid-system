"""Creator response cascade and downstream Feed value."""

from __future__ import annotations

import torch

from .world import deterministic_gumbel, deterministic_uniform, hidden_utility


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


def _cascade(world, prompt, selected_utility, clicked):
    requests, config = world.requests, world.config
    create_probability = torch.sigmoid(
        -1.35 + 0.95 * selected_utility + 0.70 * requests.activity
        - 0.55 * requests.fatigue
    )
    created = clicked & (
        deterministic_uniform(requests.request_id, prompt, 203, config.seed)
        < create_probability
    )
    quality = _quality_potential(world, prompt)
    publish_probability = torch.sigmoid(
        -0.80 + 1.1 * quality + 0.65 * requests.activity
        - 0.60 * world.catalog.difficulty[prompt]
    )
    published = created & (
        deterministic_uniform(requests.request_id, prompt, 204, config.seed)
        < publish_probability
    )
    return created, published, quality


def _training_labels(world, candidates, top, selected_rank, states):
    clicked, created, published = states
    config = world.config
    labels = torch.zeros(
        config.requests, config.merged_candidates, 4, device=top.device
    )
    batch = torch.arange(config.requests, device=top.device)
    selected_index = top.gather(1, selected_rank[:, None]).squeeze(1)
    labels[batch[clicked], selected_index[clicked], 0] = 1.0
    labels[batch[created], selected_index[created], 1] = 1.0
    labels[batch[published], selected_index[published], 2] = 1.0
    labels[:, :, 3] = (
        _candidate_quality_labels(world, candidates.prompt_ids) > 0.65
    ).float()
    return labels


def simulate_response(world, candidates, scores):
    config, requests = world.config, world.requests
    top = torch.topk(scores, config.exposed_candidates, dim=1).indices
    exposed = candidates.prompt_ids.gather(1, top)
    utility = hidden_utility(world, exposed)
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
    created, published, quality = _cascade(
        world, prompt, selected_utility, clicked
    )
    risk = torch.sigmoid(
        -4.2 + 0.9 * world.catalog.saturation[prompt] - 1.5 * quality
    )
    stay = published.float() * (
        0.70 + 2.0 * quality
        + 0.55 * (1.0 - world.catalog.saturation[prompt])
    )
    active = published.float() * torch.clamp(
        0.0012 * quality + 0.0007 * requests.activity, max=0.003
    )
    labels = _training_labels(
        world, candidates, top, selected_rank, (clicked, created, published)
    )
    return {
        "top_indices": top,
        "clicked": clicked,
        "created": created,
        "published": published,
        "quality_potential": quality,
        "content_risk": risk,
        "feed_stay_seconds": stay,
        "feed_active_day": active,
        "labels": labels,
    }
