"""Latent creator state and prompt catalog for Feed-posting simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from ...simulation.randomness import normal as counter_normal
from ...simulation.randomness import uniform as counter_uniform
from ...simulation.randomness import uniform_for_items
from ..contracts import FeedPostingConfig
from .teacher import NeuralFeedSupplyTeacher, build_neural_feed_supply_teacher


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
    cohort: torch.Tensor
    creator_id: torch.Tensor
    request_step: torch.Tensor


@dataclass(frozen=True)
class FeedPostingWorld:
    config: FeedPostingConfig
    catalog: PromptCatalog
    requests: CreatorRequests
    category_basis: torch.Tensor
    teacher: NeuralFeedSupplyTeacher | None


def normalize(values):
    return functional.normalize(values, dim=-1)


def deterministic_uniform(request_id, prompt_id, stream, seed):
    return uniform_for_items(request_id, prompt_id, 0, stream, seed)


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
        1.3 * (sequence * basis[primary, None, :]).sum(dim=2)
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
        experience, fatigue, activity, outside, torch.zeros_like(request_id),
        request_id, torch.zeros_like(request_id),
    )


def _creator_traits(config, device):
    creator = torch.arange(config.creators, device=device)
    primary = torch.floor(
        counter_uniform(creator, 0, 310, config.seed) * config.categories
    ).long()
    secondary = torch.floor(
        counter_uniform(creator, 0, 312, config.seed) * config.categories
    ).long()
    country = torch.floor(
        counter_uniform(creator, 0, 314, config.seed) * config.countries
    ).long()
    experience = torch.sigmoid(counter_normal(creator, 0, 316, config.seed))
    fatigue = torch.sigmoid(counter_normal(creator, 0, 318, config.seed))
    cohort = torch.floor(
        counter_uniform(creator, 0, 319, config.seed) * 8
    ).long()
    return primary, secondary, country, experience, fatigue, cohort


def _creator_sequence_chunk(config, request_id, basis, primary, secondary):
    positions = torch.arange(config.sequence_length, device=request_id.device)
    switch = counter_uniform(
        request_id, 0, 320, config.seed, config.sequence_length
    ) < (0.20 + 0.25 * positions / config.sequence_length)
    random_category = torch.floor(
        counter_uniform(
            request_id, 0, 322, config.seed, config.sequence_length
        ) * config.categories
    ).long()
    categories = torch.where(
        switch, random_category,
        torch.where(
            positions[None, :] % 4 == 0,
            secondary[:, None], primary[:, None],
        ),
    )
    noise = counter_normal(
        request_id, 0, 324, config.seed,
        config.sequence_length * config.semantic_dim,
    ).reshape(-1, config.sequence_length, config.semantic_dim)
    sequence = normalize(basis[categories] + 0.42 * noise)
    feedback = torch.sigmoid(
        1.3 * (sequence * basis[primary, None, :]).sum(dim=2)
        + 0.8 * counter_normal(
            request_id, 0, 326, config.seed, config.sequence_length
        )
    )
    recency = torch.linspace(
        0.25, 1.0, config.sequence_length, device=request_id.device
    )[None, :]
    weight = feedback * recency
    summary = normalize(
        (sequence * weight[:, :, None]).sum(dim=1)
        / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    )
    return sequence, feedback, summary, categories[:, -1]


def _creator_sequence(config, request_id, basis, primary, secondary):
    outputs = ([], [], [], [])
    batch = config.generation_batch_requests
    for start in range(0, len(request_id), batch):
        values = _creator_sequence_chunk(
            config,
            request_id[start : start + batch],
            basis,
            primary[start : start + batch],
            secondary[start : start + batch],
        )
        for destination, value in zip(outputs, values, strict=True):
            destination.append(value)
    return tuple(torch.cat(values) for values in outputs)


def _build_creator_requests_v4(
    config, device, basis, request_start=0, request_count=None,
):
    count = config.requests if request_count is None else request_count
    request_id = torch.arange(
        request_start, request_start + count, device=device
    )
    creator_id = torch.remainder(request_id, config.creators)
    request_step = torch.div(request_id, config.creators, rounding_mode="floor")
    primary, secondary, country, experience, fatigue, cohort = _creator_traits(
        config, device
    )
    creator_primary = primary[creator_id]
    creator_secondary = secondary[creator_id]
    drift = torch.sigmoid((request_step.float() - 3.0) * 0.8)[:, None]
    latent_intent = normalize(
        basis[creator_primary]
        + (0.18 + 0.30 * drift) * basis[creator_secondary]
        + 0.28 * counter_normal(
            request_id, 0, 328, config.seed, config.semantic_dim
        )
    )
    observed_profile = normalize(
        latent_intent + 0.50 * counter_normal(
            request_id, 0, 330, config.seed, config.semantic_dim
        )
    )
    sequence, feedback, summary, recent_category = _creator_sequence(
        config, request_id, basis, creator_primary, creator_secondary
    )
    creator_category = torch.where(
        counter_uniform(request_id, 0, 332, config.seed) < 0.70,
        creator_primary, creator_secondary,
    )
    creator_history = normalize(
        basis[creator_category] + 0.45 * counter_normal(
            request_id, 0, 334, config.seed, config.semantic_dim
        )
    )
    request_experience = torch.clamp(
        experience[creator_id] + 0.03 * request_step.float(), 0.0, 1.0
    )
    request_fatigue = torch.clamp(
        fatigue[creator_id] + 0.04 * request_step.float()
        + 0.08 * counter_normal(request_id, 0, 336, config.seed),
        0.0, 1.0,
    )
    activity = torch.sigmoid(
        0.7 * request_experience - 0.5 * request_fatigue
        + counter_normal(request_id, 0, 338, config.seed)
    )
    outside = (
        0.55 - 0.80 * activity + 0.55 * request_fatigue
        + 0.35 * counter_normal(request_id, 0, 340, config.seed)
    )
    return CreatorRequests(
        request_id, latent_intent, observed_profile, sequence, feedback,
        summary, recent_category, creator_history, creator_category,
        country[creator_id], request_experience, request_fatigue, activity,
        outside, cohort[creator_id], creator_id, request_step,
    )


def build_world(config: FeedPostingConfig):
    device = torch.device(config.device)
    catalog_seed = config.seed if config.catalog_seed is None else config.catalog_seed
    generator = torch.Generator(device=device).manual_seed(catalog_seed)
    basis = normalize(torch.randn(
        config.categories, config.semantic_dim,
        generator=generator, device=device,
    ))
    catalog = _build_catalog(config, generator, device, basis)
    if config.world_version == "creator-neural-feed-supply-v4":
        return FeedPostingWorld(
            config, catalog,
            _build_creator_requests_v4(config, device, basis), basis,
            build_neural_feed_supply_teacher(device),
        )
    return FeedPostingWorld(
        config, catalog, _build_requests(config, generator, device, basis),
        basis, None,
    )


def build_world_partition(config, request_start, request_count):
    if config.world_version != "creator-neural-feed-supply-v4":
        raise ValueError("partitioned Feed posting requires creator V4")
    if request_start < 0 or request_count < 1:
        raise ValueError("invalid Feed posting request partition")
    if request_start + request_count > config.requests:
        raise ValueError("Feed posting request partition exceeds world")
    device = torch.device(config.device)
    catalog_seed = config.seed if config.catalog_seed is None else config.catalog_seed
    generator = torch.Generator(device=device).manual_seed(catalog_seed)
    basis = normalize(torch.randn(
        config.categories, config.semantic_dim,
        generator=generator, device=device,
    ))
    return FeedPostingWorld(
        config,
        _build_catalog(config, generator, device, basis),
        _build_creator_requests_v4(
            config, device, basis, request_start, request_count
        ),
        basis,
        build_neural_feed_supply_teacher(device),
    )


def hidden_utility(world, prompt_ids):
    if world.teacher is not None:
        return hidden_feed_outputs(world, prompt_ids)[:, :, 0]
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


def hidden_feed_outputs(world, prompt_ids):
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
    inputs = torch.stack((
        intent, sequence, creator, difficulty, country,
        catalog.trend[prompt_ids], catalog.quality[prompt_ids],
        catalog.saturation[prompt_ids],
        requests.experience[:, None].expand_as(intent),
        requests.fatigue[:, None].expand_as(intent),
        requests.activity[:, None].expand_as(intent),
        requests.request_step[:, None].float().expand_as(intent) / 8.0,
        torch.sin(requests.cohort[:, None].float()).expand_as(intent),
        torch.cos(requests.cohort[:, None].float()).expand_as(intent),
        (requests.country[:, None].float() / world.config.countries).expand_as(intent),
    ), dim=2)
    return world.teacher(inputs)
