"""Private ecosystem state; no platform package may import this module."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch

from ...randomness.counter import normal, uniform
from ..catalog import PublicCatalog
from ..contracts import ContentKind
from .dynamics.population import POPULATION_VERSION, sample_population
from .dynamics.growth import sample_acquisition_population
from .dynamics.needs import sample_need_population


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
    initialization_mode: Literal["equilibrium", "bootstrap"] = "equilibrium"

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
        if self.initialization_mode not in {"equilibrium", "bootstrap"}:
            raise ValueError("world initialization mode is unsupported")


@dataclass
class HiddenUserState:
    user_id: torch.Tensor
    creator_id: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    timezone_offset: torch.Tensor
    language: torch.Tensor
    device_class: torch.Tensor
    lifecycle_cohort: torch.Tensor
    lifecycle_stage: torch.Tensor
    acquisition_channel: torch.Tensor
    acquisition_quality: torch.Tensor
    referral_susceptibility: torch.Tensor
    weekly_activity: torch.Tensor
    churn_susceptibility: torch.Tensor
    segment: torch.Tensor
    primary_topic: torch.Tensor
    secondary_topic: torch.Tensor
    long_interest: torch.Tensor
    short_interest: torch.Tensor
    behavior_sequence: torch.Tensor
    exposure_item: torch.Tensor
    exposure_creator: torch.Tensor
    exposure_topic: torch.Tensor
    exposure_time: torch.Tensor
    exposure_positive: torch.Tensor
    exposure_cursor: torch.Tensor
    disappointment: torch.Tensor
    search_followup_topic: torch.Tensor
    search_reformulation_depth: torch.Tensor
    post_search_feed_pending: torch.Tensor
    last_search_query_id: torch.Tensor
    surface_intent: torch.Tensor
    response_style: torch.Tensor
    satisfaction: torch.Tensor
    fatigue: torch.Tensor
    habit: torch.Tensor
    activity: torch.Tensor
    novelty: torch.Tensor
    spending_power: torch.Tensor
    need_kind: torch.Tensor
    need_topic: torch.Tensor
    need_strength: torch.Tensor
    need_expiry_time: torch.Tensor
    activation_score: torch.Tensor
    session_value_ema: torch.Tensor
    last_active_time: torch.Tensor
    last_active_day: torch.Tensor
    return_streak: torch.Tensor
    signup_time: torch.Tensor
    next_return_time: torch.Tensor
    reactivation_time: torch.Tensor
    registered: torch.Tensor
    churned: torch.Tensor
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
    trend_strength: torch.Tensor
    population_version: str = POPULATION_VERSION


@dataclass(frozen=True)
class RequestStateOverride:
    """Request-local potential outcome inputs; never committed to the world."""

    long_interest: torch.Tensor | None = None
    short_interest: torch.Tensor | None = None
    behavior_sequence: torch.Tensor | None = None
    fatigue: torch.Tensor | None = None
    satisfaction: torch.Tensor | None = None
    public_quality: torch.Tensor | None = None
    hidden_quality: torch.Tensor | None = None


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


def _empty_search_state(user: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "search_followup_topic": torch.full_like(user, -1),
        "search_reformulation_depth": torch.zeros_like(user),
        "post_search_feed_pending": torch.zeros_like(user, dtype=torch.bool),
        "last_search_query_id": torch.full_like(user, -1),
    }


def _initial_lifecycle_state(
    config: UserWorldConfig,
    user: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    future_signup = (
        uniform(user, 0, 1_057, config.environment_seed)
        < config.future_signup_fraction
    )
    future_signup_time = 1 + torch.floor(
        config.ticks_per_day
        * 28
        * uniform(user, 0, 1_061, config.environment_seed)
    ).long()
    if config.initialization_mode == "bootstrap":
        signup_time = torch.where(
            future_signup, future_signup_time, torch.zeros_like(user),
        )
        return signup_time, signup_time.clone(), torch.zeros_like(
            user, dtype=torch.bool,
        )
    account_age = 1 + torch.floor(
        config.ticks_per_day
        * 365
        * uniform(user, 0, 1_063, config.environment_seed)
    ).long()
    residual_arrival = torch.floor(
        config.ticks_per_day
        * uniform(user, 0, 1_067, config.environment_seed)
    ).long()
    return (
        torch.where(future_signup, future_signup_time, -account_age),
        torch.where(future_signup, future_signup_time, residual_arrival),
        ~future_signup,
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


def _initial_dynamic_context(config, user, population, signup_time, registered):
    acquisition = sample_acquisition_population(
        user,
        population.country,
        population.mixture,
        population.habit,
        config.environment_seed,
    )
    needs = sample_need_population(
        user,
        population.primary_topic,
        population.secondary_topic,
        population.surface_intent,
        config.ticks_per_day,
        config.topics,
        config.environment_seed,
    )
    account_age_days = (-signup_time).clamp_min(0).float() / config.ticks_per_day
    sessions = torch.floor(
        account_age_days.sqrt() * (0.8 + 6.0 * population.activity)
    ).long()
    activation = torch.sigmoid(
        -1.4
        + 1.2 * acquisition.quality
        + population.habit
        + 0.22 * torch.log1p(sessions.float())
    )
    stage = torch.where(
        sessions >= 30,
        torch.full_like(user, 3),
        torch.where(
            sessions >= 5,
            torch.full_like(user, 2),
            torch.where(registered, torch.ones_like(user), torch.zeros_like(user)),
        ),
    )
    return {
        "lifecycle_stage": stage,
        "acquisition_channel": acquisition.channel,
        "acquisition_quality": acquisition.quality,
        "referral_susceptibility": acquisition.referral_susceptibility,
        "need_kind": needs.kind,
        "need_topic": needs.topic,
        "need_strength": needs.strength,
        "need_expiry_time": needs.expiry_time,
        "activation_score": activation,
        "session_count": sessions,
    }


def _empty_behavior_memory(config, user, device):
    return {
        "behavior_sequence": torch.zeros(
            config.users, 24, 8, device=device, dtype=torch.float16,
        ),
        "exposure_item": torch.full(
            (config.users, 64), -1, device=device, dtype=torch.long,
        ),
        "exposure_creator": torch.full(
            (config.users, 64), -1, device=device, dtype=torch.long,
        ),
        "exposure_topic": torch.full(
            (config.users, 64), -1, device=device, dtype=torch.long,
        ),
        "exposure_time": torch.full(
            (config.users, 64), -1, device=device, dtype=torch.long,
        ),
        "exposure_positive": torch.zeros(
            (config.users, 64), device=device, dtype=torch.bool,
        ),
        "exposure_cursor": torch.zeros_like(user),
        "disappointment": torch.zeros(config.users, device=device),
        "session_value_ema": torch.zeros(config.users, device=device),
        "last_active_time": torch.full_like(user, -1),
        "last_active_day": torch.full_like(user, -1),
        "return_streak": torch.zeros_like(user),
    }


def build_hidden_users(
    config: UserWorldConfig,
    catalog: PublicCatalog,
) -> HiddenUserState:
    device = catalog.item_id.device
    user = torch.arange(config.users, device=device)
    prototype = topic_prototypes(catalog, config.topics)
    population = sample_population(
        user,
        topics=config.topics,
        countries=config.countries,
        regions_per_country=config.regions_per_country,
        seed=config.environment_seed,
    )
    primary = population.primary_topic
    secondary = population.secondary_topic
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
    country = population.country
    region = population.region
    segment = population.mixture
    signup_time, next_return_time, registered = _initial_lifecycle_state(
        config, user,
    )
    dynamic = _initial_dynamic_context(
        config, user, population, signup_time, registered,
    )
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
        timezone_offset=population.timezone_offset,
        language=population.language,
        device_class=population.device_class,
        lifecycle_cohort=population.lifecycle_cohort,
        **dynamic,
        weekly_activity=population.weekly_activity,
        churn_susceptibility=population.churn_susceptibility,
        segment=segment,
        primary_topic=primary,
        secondary_topic=secondary,
        long_interest=long_interest,
        short_interest=short_interest,
        **_empty_behavior_memory(config, user, device),
        **_empty_search_state(user),
        surface_intent=population.surface_intent,
        response_style=population.response_style,
        satisfaction=population.satisfaction,
        fatigue=population.fatigue,
        habit=population.habit,
        activity=population.activity,
        novelty=population.novelty,
        spending_power=population.spending_power,
        signup_time=signup_time,
        next_return_time=next_return_time,
        reactivation_time=torch.full_like(
            user, torch.iinfo(torch.long).max // 4,
        ),
        registered=registered,
        churned=torch.zeros(config.users, device=device, dtype=torch.bool),
        active=torch.zeros(config.users, device=device, dtype=torch.bool),
        session_depth=torch.zeros(config.users, device=device, dtype=torch.long),
    )
