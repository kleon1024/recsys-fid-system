"""Hidden creator population, response, retention, and catalog refresh."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import torch

from ...simulation.randomness import normal, uniform


@dataclass
class CreatorPopulation:
    creator_ids: torch.Tensor
    mixture: torch.Tensor
    region: torch.Tensor
    active: torch.Tensor
    motivation: torch.Tensor
    fatigue: torch.Tensor
    quality: torch.Tensor
    topic: torch.Tensor
    expected_exposure: torch.Tensor
    cumulative_posts: torch.Tensor
    cumulative_retained_days: torch.Tensor


@dataclass
class CreatorFeedback:
    exposures: torch.Tensor
    stay: torch.Tensor
    long_view: torch.Tensor
    engagement: torch.Tensor
    negative: torch.Tensor

    @classmethod
    def empty(cls, creators, device):
        return cls(*(
            torch.zeros(creators, device=device) for _ in range(5)
        ))

    def add(self, author, values, active):
        index = author[active]
        self.exposures.scatter_add_(0, index, torch.ones_like(index).float())
        fields = (
            (self.stay, values["stay"]),
            (self.long_view, values["long_view"].float()),
            (self.engagement, (
                values["like"].float() + values["comment"].float()
                + values["share"].float() + values["follow"].float()
            )),
            (self.negative, values["negative"].float()),
        )
        for target, value in fields:
            target.scatter_add_(0, index, value[active])


def initialize_creators(catalog, creators, seed):
    device = catalog.quality.device
    ids = torch.arange(creators, device=device)
    counts = torch.zeros(creators, device=device)
    quality = torch.zeros(creators, device=device)
    topics = torch.zeros(creators, catalog.topics.shape[1], device=device)
    counts.scatter_add_(0, catalog.author, torch.ones_like(catalog.quality))
    quality.scatter_add_(0, catalog.author, catalog.quality)
    topics.index_add_(0, catalog.author, catalog.topics)
    quality = quality / counts.clamp_min(1.0)
    topics = torch.nn.functional.normalize(topics + 0.01, dim=1)
    return CreatorPopulation(
        creator_ids=ids,
        mixture=torch.floor(uniform(ids, 0, 401, seed) * 4).long(),
        region=torch.floor(uniform(ids, 0, 403, seed) * 10).long(),
        active=torch.ones(creators, dtype=torch.bool, device=device),
        motivation=0.35 + 0.55 * uniform(ids, 0, 405, seed),
        fatigue=0.10 * uniform(ids, 0, 407, seed),
        quality=quality,
        topic=topics,
        expected_exposure=(4.0 + 12.0 * counts.sqrt()),
        cumulative_posts=counts.clone(),
        cumulative_retained_days=torch.ones(creators, device=device),
    )


class CreatorResponseWorld:
    """Hidden mixture-of-experts transition for provider behavior."""

    def __init__(self, device, seed, inputs=12, width=48):
        generator = torch.Generator(device=device).manual_seed(seed + 190_001)
        self.input_weight = torch.randn(
            inputs, width, generator=generator, device=device
        ) / sqrt(inputs)
        self.input_bias = 0.10 * torch.randn(
            width, generator=generator, device=device
        )
        self.expert_weight = 0.55 * torch.randn(
            4, width, 4, generator=generator, device=device
        ) / sqrt(width)
        self.expert_bias = torch.randn(
            4, 4, generator=generator, device=device
        ) * 0.05

    def _outputs(self, population, feedback, day):
        exposure = feedback.exposures
        engagement = feedback.engagement / exposure.clamp_min(1.0)
        negative = feedback.negative / exposure.clamp_min(1.0)
        stay = feedback.stay / exposure.clamp_min(1.0) / 30.0
        surprise = torch.tanh(
            torch.log1p(exposure) - torch.log1p(population.expected_exposure)
        )
        inputs = torch.stack((
            population.motivation, population.fatigue, population.quality,
            surprise, stay, engagement, negative,
            population.region.float() / 9.0,
            population.mixture.float() / 3.0,
            torch.full_like(exposure, day / 14.0),
            torch.log1p(population.cumulative_posts) / 8.0,
            population.active.float(),
        ), dim=1)
        hidden = torch.nn.functional.silu(
            inputs @ self.input_weight + self.input_bias
        )
        hidden += 0.30 * torch.sin(hidden * torch.roll(hidden, 5, dims=1))
        expert = torch.einsum("bd,edk->bek", hidden, self.expert_weight)
        expert += self.expert_bias[None]
        rows = torch.arange(len(inputs), device=inputs.device)
        return expert[rows, population.mixture], surprise, engagement, negative

    def advance(self, population, feedback, day, seed, max_new_items):
        outputs, surprise, engagement, negative = self._outputs(
            population, feedback, day
        )
        publish_probability = torch.sigmoid(
            -2.7 + 1.2 * population.motivation - 0.8 * population.fatigue
            + 0.7 * surprise + outputs[:, 0]
        ) * population.active
        retain_probability = torch.sigmoid(
            2.4 + 0.8 * population.motivation - 1.0 * population.fatigue
            + 0.4 * surprise - 0.8 * negative + outputs[:, 1]
        )
        published = (
            uniform(population.creator_ids, day, 421, seed) < publish_probability
        ) & population.active
        retained = (
            uniform(population.creator_ids, day, 423, seed) < retain_probability
        ) & population.active
        population.active = retained
        population.motivation = torch.clamp(
            0.82 * population.motivation + 0.10 * engagement
            + 0.10 * surprise - 0.12 * negative + 0.05 * outputs[:, 2],
            0.0, 1.0,
        )
        population.fatigue = torch.clamp(
            0.78 * population.fatigue + 0.08 * published.float()
            + 0.12 * negative - 0.04 * surprise, 0.0, 1.0,
        )
        population.quality = torch.sigmoid(
            torch.logit(population.quality.clamp(0.02, 0.98))
            + 0.08 * outputs[:, 3] + 0.05 * engagement
        )
        population.expected_exposure = (
            0.85 * population.expected_exposure + 0.15 * feedback.exposures
        ).clamp_min(1.0)
        population.cumulative_posts += published
        population.cumulative_retained_days += retained
        publishers = torch.nonzero(published).flatten()[:max_new_items]
        return publishers


def refresh_catalog(catalog, population, publishers, day, seed, behavior_world):
    need_by_creator = population.active.float() * torch.clamp(
        0.55 * (1.0 - population.motivation)
        + 0.30 * population.fatigue
        + 0.15 / torch.sqrt(population.expected_exposure.clamp_min(1.0)),
        0.0, 1.0,
    )
    creator_need = need_by_creator[catalog.author]
    if not len(publishers):
        return replace(catalog, creator_need=creator_need)
    replace_score = catalog.freshness + 0.25 * catalog.popularity
    slots = torch.topk(-replace_score, len(publishers)).indices
    noise = normal(slots, day, 431, seed, catalog.topics.shape[1])
    topics = catalog.topics.clone()
    topics[slots] = torch.nn.functional.normalize(
        population.topic[publishers] + 0.18 * noise, dim=1
    )
    category = catalog.category.clone()
    category[slots] = topics[slots].argmax(dim=1)
    quality = catalog.quality.clone()
    quality[slots] = population.quality[publishers]
    freshness = torch.clamp(catalog.freshness - 0.04, 0.0, 1.0)
    freshness[slots] = 1.0
    popularity = catalog.popularity.clone()
    popularity[slots] = 0.01
    author = catalog.author.clone()
    author[slots] = publishers
    content_type = catalog.content_type.clone()
    content_type[slots] = 0
    updated = replace(
        catalog, topics=topics, category=category, quality=quality,
        freshness=freshness, popularity=popularity, author=author,
        content_type=content_type, creator_need=need_by_creator[author],
    )
    return behavior_world.decorate_new_supply(
        updated, slots, category[slots], day
    )
