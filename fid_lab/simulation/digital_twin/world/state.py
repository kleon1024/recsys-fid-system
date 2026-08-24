"""Private ecosystem state; no platform package may import this module."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ...randomness.counter import normal, uniform
from ..catalog import PublicCatalog
from ..contracts import ContentKind, Surface


@dataclass(frozen=True)
class UserWorldConfig:
    users: int
    topics: int
    embedding_dim: int
    countries: int
    regions_per_country: int
    environment_seed: int
    ticks_per_day: int = 96
    future_signup_fraction: float = 0.08

    def __post_init__(self):
        integer_fields = (
            self.users,
            self.topics,
            self.embedding_dim,
            self.countries,
            self.regions_per_country,
            self.ticks_per_day,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("world dimensions must be positive")
        if not 0.0 <= self.future_signup_fraction < 1.0:
            raise ValueError("future signup fraction must be in [0, 1)")


@dataclass
class HiddenUserState:
    user_id: torch.Tensor
    creator_id: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    timezone_offset: torch.Tensor
    segment: torch.Tensor
    primary_topic: torch.Tensor
    secondary_topic: torch.Tensor
    long_interest: torch.Tensor
    short_interest: torch.Tensor
    surface_intent: torch.Tensor
    response_style: torch.Tensor
    satisfaction: torch.Tensor
    fatigue: torch.Tensor
    habit: torch.Tensor
    activity: torch.Tensor
    novelty: torch.Tensor
    spending_power: torch.Tensor
    signup_time: torch.Tensor
    next_return_time: torch.Tensor
    registered: torch.Tensor
    active: torch.Tensor
    session_depth: torch.Tensor
    session_count: torch.Tensor

    def clone(self) -> HiddenUserState:
        return HiddenUserState(**{
            field.name: getattr(self, field.name).clone()
            for field in fields(self)
        })


@dataclass(frozen=True)
class HiddenCatalogTruth:
    semantic_embedding: torch.Tensor
    quality: torch.Tensor
    risk: torch.Tensor
    price_appeal: torch.Tensor


@dataclass(frozen=True)
class UserWorldSnapshot:
    users: HiddenUserState
    catalog_truth: HiddenCatalogTruth
    ticks_per_day: int
    environment_seed: int
    item_creator_id: torch.Tensor
    item_product_id: torch.Tensor
    item_poi_id: torch.Tensor
    item_country: torch.Tensor
    item_region: torch.Tensor
    item_publish_time: torch.Tensor


def topic_prototypes(catalog: PublicCatalog, topics: int) -> torch.Tensor:
    dimensions = catalog.content_embedding.shape[1]
    result = torch.zeros(
        topics, dimensions, device=catalog.item_id.device,
    )
    counts = torch.zeros(topics, device=catalog.item_id.device)
    result.index_add_(0, catalog.topic_id, catalog.content_embedding)
    counts.index_add_(0, catalog.topic_id, torch.ones_like(counts[catalog.topic_id]))
    return torch.nn.functional.normalize(
        result / counts.clamp_min(1.0)[:, None], dim=1,
    )


def build_hidden_catalog_truth(
    catalog: PublicCatalog, environment_seed: int,
) -> HiddenCatalogTruth:
    """Generate truth causally from public attributes plus hidden residuals."""
    item = catalog.item_id
    residual = normal(
        item, 0, 1_001, environment_seed,
        catalog.content_embedding.shape[1],
    )
    semantic = torch.nn.functional.normalize(
        catalog.content_embedding + 0.42 * residual, dim=1,
    )
    prior = catalog.quality_prior.clamp(1e-4, 1.0 - 1e-4)
    quality_logit = torch.logit(prior)
    quality = torch.sigmoid(
        quality_logit + 0.75 * normal(item, 0, 1_009, environment_seed)
    )
    risk = torch.sigmoid(
        -3.2 + 0.55 * (1.0 - quality)
        + 1.15 * normal(item, 0, 1_013, environment_seed)
    )
    price_appeal = torch.sigmoid(
        1.1 - 0.45 * torch.log1p(catalog.price)
        + 0.8 * normal(item, 0, 1_019, environment_seed)
    )
    return HiddenCatalogTruth(semantic, quality, risk, price_appeal)


def build_hidden_users(
    config: UserWorldConfig,
    catalog: PublicCatalog,
) -> HiddenUserState:
    device = catalog.item_id.device
    user = torch.arange(config.users, device=device)
    prototype = topic_prototypes(catalog, config.topics)
    primary = torch.remainder(user * 503 + 19, config.topics)
    secondary = torch.remainder(user * 1_009 + 31, config.topics)
    residual = normal(
        user, 0, 1_031, config.environment_seed, config.embedding_dim,
    )
    long_interest = torch.nn.functional.normalize(
        0.85 * prototype[primary]
        + 0.45 * prototype[secondary]
        + 0.55 * residual,
        dim=1,
    )
    short_interest = torch.nn.functional.normalize(
        long_interest
        + 0.48 * normal(
            user, 0, 1_039, config.environment_seed, config.embedding_dim,
        ),
        dim=1,
    )
    country = torch.remainder(user * 101 + 7, config.countries)
    region = country * config.regions_per_country + torch.remainder(
        user * 307 + 11, config.regions_per_country,
    )
    segment = torch.remainder(user * 61 + country * 7, 12)
    surface_logits = normal(
        user, 0, 1_049, config.environment_seed, len(Surface),
    )
    surface_logits[:, int(Surface.FEED)] += 1.4
    surface_logits[:, int(Surface.SEARCH)] += 0.25
    surface_intent = torch.softmax(surface_logits, dim=1)
    future = uniform(user, 0, 1_057, config.environment_seed)
    signup_time = torch.where(
        future < config.future_signup_fraction,
        1 + torch.floor(
            config.ticks_per_day
            * 14 * uniform(user, 0, 1_061, config.environment_seed)
        ).long(),
        torch.zeros_like(user),
    )
    base = uniform(user, 0, 1_063, config.environment_seed)
    post_kind = (
        (catalog.content_kind == int(ContentKind.SHORT_VIDEO))
        | (catalog.content_kind == int(ContentKind.PHOTO))
        | (catalog.content_kind == int(ContentKind.ARTICLE))
        | (catalog.content_kind == int(ContentKind.CARD))
    )
    reserved_creators = torch.unique(
        catalog.creator_id[~catalog.active & post_kind], sorted=True,
    )
    if not len(reserved_creators):
        reserved_creators = torch.unique(catalog.creator_id, sorted=True)
    creator_id = reserved_creators[
        torch.remainder(user * 503 + 17, len(reserved_creators))
    ]
    return HiddenUserState(
        user_id=user,
        creator_id=creator_id,
        country=country,
        region=region,
        timezone_offset=(country * 3 + segment).remainder(24) - 12,
        segment=segment,
        primary_topic=primary,
        secondary_topic=secondary,
        long_interest=long_interest,
        short_interest=short_interest,
        surface_intent=surface_intent,
        response_style=normal(
            user, 0, 1_069, config.environment_seed, 8,
        ).clamp(-2.5, 2.5),
        satisfaction=(0.38 + 0.34 * base).clamp(0.0, 1.0),
        fatigue=0.05 + 0.18 * uniform(
            user, 0, 1_073, config.environment_seed,
        ),
        habit=0.12 + 0.72 * uniform(
            user, 0, 1_079, config.environment_seed,
        ),
        activity=0.03 + 0.62 * uniform(
            user, 0, 1_081, config.environment_seed,
        ).square(),
        novelty=uniform(user, 0, 1_087, config.environment_seed),
        spending_power=uniform(user, 0, 1_091, config.environment_seed),
        signup_time=signup_time,
        next_return_time=signup_time.clone(),
        registered=torch.zeros(config.users, device=device, dtype=torch.bool),
        active=torch.zeros(config.users, device=device, dtype=torch.bool),
        session_depth=torch.zeros(config.users, device=device, dtype=torch.long),
        session_count=torch.zeros(config.users, device=device, dtype=torch.long),
    )
