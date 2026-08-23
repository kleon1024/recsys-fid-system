"""Latent creator state and prompt catalog for Feed-posting simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from ..contracts import FeedPostingConfig


@dataclass(frozen=True)
class PromptCatalog:
    category: torch.Tensor
    semantic: torch.Tensor
    trend: torch.Tensor
    quality: torch.Tensor
    difficulty: torch.Tensor
    saturation: torch.Tensor
    country: torch.Tensor


@dataclass(frozen=True)
class CreatorRequests:
    request_id: torch.Tensor
    latent_intent: torch.Tensor
    observed_profile: torch.Tensor
    feed_sequence: torch.Tensor
    sequence_feedback: torch.Tensor
    sequence_summary: torch.Tensor
    recent_category: torch.Tensor
    creator_history: torch.Tensor
    creator_category: torch.Tensor
    country: torch.Tensor
    experience: torch.Tensor
    fatigue: torch.Tensor
    activity: torch.Tensor
    outside_preference: torch.Tensor


@dataclass(frozen=True)
class FeedPostingWorld:
    config: FeedPostingConfig
    catalog: PromptCatalog
    requests: CreatorRequests
    category_basis: torch.Tensor


def normalize(values):
    return functional.normalize(values, dim=-1)


def deterministic_uniform(request_id, prompt_id, stream, seed):
    value = torch.remainder(
        request_id.long() * 1_103_515_245
        + prompt_id.long() * 48_271
        + stream * 7_919
        + seed * 503,
        2**31 - 1,
    )
    return (value.float() + 0.5) / float(2**31 - 1)


def deterministic_gumbel(request_id, prompt_id, stream, seed):
    draw = deterministic_uniform(
        request_id, prompt_id, stream, seed
    ).clamp(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(draw))


def category_prompt(config, category, offset):
    per_category = config.prompts // config.categories
    return category * per_category + torch.remainder(offset, per_category)


def _build_catalog(config, generator, device, basis):
    prompt = torch.arange(config.prompts, device=device)
    category = torch.remainder(prompt, config.categories)
    semantic = normalize(
        basis[category] + 0.34 * torch.randn(
            config.prompts, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    trend = torch.sigmoid(
        1.2 * torch.randn(config.prompts, generator=generator, device=device)
    )
    quality = torch.sigmoid(
        torch.randn(config.prompts, generator=generator, device=device)
    )
    difficulty = torch.sigmoid(
        torch.randn(config.prompts, generator=generator, device=device)
    )
    saturation = torch.sigmoid(
        0.8 * trend + 0.8 * torch.randn(
            config.prompts, generator=generator, device=device
        )
    )
    country = torch.remainder(prompt * 7 + 3, config.countries)
    return PromptCatalog(
        category, semantic, trend, quality, difficulty, saturation, country
    )


def _sequence_state(config, generator, device, basis, primary, secondary):
    positions = torch.arange(config.sequence_length, device=device)[None, :]
    switch = torch.rand(
        config.requests, config.sequence_length,
        generator=generator, device=device,
    ) < (0.20 + 0.25 * positions / config.sequence_length)
    random_category = torch.randint(
        config.categories, (config.requests, config.sequence_length),
        generator=generator, device=device,
    )
    categories = torch.where(
        switch, random_category,
        torch.where(positions % 4 == 0, secondary[:, None], primary[:, None]),
    )
    sequence = normalize(
        basis[categories] + 0.42 * torch.randn(
            config.requests, config.sequence_length, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    feedback = torch.sigmoid(
        1.3 * torch.einsum("bld,bd->bl", sequence, basis[primary])
        + 0.8 * torch.randn(
            config.requests, config.sequence_length,
            generator=generator, device=device,
        )
    )
    recency = torch.linspace(
        0.25, 1.0, config.sequence_length, device=device
    )[None, :]
    weight = feedback * recency
    summary = normalize(
        (sequence * weight[:, :, None]).sum(dim=1)
        / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    )
    return sequence, feedback, summary, categories[:, -1]


def _build_requests(config, generator, device, basis):
    request_id = torch.arange(config.requests, device=device)
    primary = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    secondary = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    latent_intent = normalize(
        basis[primary] + 0.24 * basis[secondary] + 0.28 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    observed_profile = normalize(
        latent_intent + 0.50 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    sequence, feedback, summary, recent_category = _sequence_state(
        config, generator, device, basis, primary, secondary
    )
    creator_category = torch.where(
        torch.rand(config.requests, generator=generator, device=device) < 0.70,
        primary, secondary,
    )
    creator_history = normalize(
        basis[creator_category] + 0.45 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    country = torch.randint(
        config.countries, (config.requests,), generator=generator, device=device
    )
    experience = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    fatigue = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    activity = torch.sigmoid(
        0.7 * experience - 0.4 * fatigue
        + torch.randn(config.requests, generator=generator, device=device)
    )
    outside = 0.55 - 0.80 * activity + 0.55 * fatigue + 0.35 * torch.randn(
        config.requests, generator=generator, device=device
    )
    return CreatorRequests(
        request_id, latent_intent, observed_profile, sequence, feedback,
        summary, recent_category, creator_history, creator_category, country,
        experience, fatigue, activity, outside,
    )


def build_world(config: FeedPostingConfig):
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    basis = normalize(torch.randn(
        config.categories, config.semantic_dim,
        generator=generator, device=device,
    ))
    return FeedPostingWorld(
        config,
        _build_catalog(config, generator, device, basis),
        _build_requests(config, generator, device, basis),
        basis,
    )


def hidden_utility(world, prompt_ids):
    catalog, requests = world.catalog, world.requests
    intent = torch.einsum(
        "bkd,bd->bk", catalog.semantic[prompt_ids], requests.latent_intent
    )
    sequence = torch.einsum(
        "bkd,bd->bk", catalog.semantic[prompt_ids], requests.sequence_summary
    )
    creator = torch.einsum(
        "bkd,bd->bk", catalog.semantic[prompt_ids], requests.creator_history
    )
    difficulty = 1.0 - (
        catalog.difficulty[prompt_ids] - requests.experience[:, None]
    ).abs()
    country = (
        catalog.country[prompt_ids] == requests.country[:, None]
    ).float()
    nonlinear = torch.sin(
        2.7 * intent + 1.5 * catalog.trend[prompt_ids]
        - 1.2 * catalog.saturation[prompt_ids]
    )
    return (
        1.45 * intent + 0.52 * sequence + 0.38 * creator
        + 0.38 * difficulty + 0.26 * country
        + 0.24 * catalog.trend[prompt_ids]
        + 0.20 * catalog.quality[prompt_ids]
        - 0.22 * catalog.saturation[prompt_ids]
        + 0.15 * nonlinear - 0.25 * requests.fatigue[:, None]
    )
