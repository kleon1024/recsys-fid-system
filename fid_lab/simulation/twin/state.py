"""Device-resident user, supply, and event-ledger state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from ..randomness.counter import normal, uniform
from .contracts import ItemKind, Surface, TwinConfig
from .environment.latent import LatentCatalogState, LatentUserState
from .platform.state import (
    CatalogState,
    ExposureLedger,
    UserState,
    select_users as select_users,
    writeback_users as writeback_users,
)
from .world.context import ContextState


@dataclass
class TwinSnapshot:
    users: tuple[UserState, ...]
    latent_users: tuple[LatentUserState, ...]
    catalog: CatalogState
    latent_catalog: LatentCatalogState
    step: int
    preperiod_user_metrics: tuple[torch.Tensor, ...]
    context: ContextState

    def fork(self):
        return TwinSnapshot(
            tuple(users.clone() for users in self.users),
            tuple(users.clone() for users in self.latent_users),
            self.catalog.clone(),
            self.latent_catalog.clone(),
            self.step,
            tuple(value.clone() for value in self.preperiod_user_metrics),
            self.context.clone(),
        )


@dataclass(frozen=True)
class _UserSeeds:
    user_id: torch.Tensor
    long_interest: torch.Tensor
    lifecycle: torch.Tensor
    country: torch.Tensor
    activity_tier: torch.Tensor
    socioeconomic: torch.Tensor
    activity_rates: torch.Tensor
    signup_step: torch.Tensor
    registered: torch.Tensor


def _initialize_ledger(config: TwinConfig, count: int, device):
    shape = (count, config.history_length)
    ledger = ExposureLedger(
        item=torch.full(shape, -1, device=device, dtype=torch.int32),
        author=torch.full(shape, -1, device=device, dtype=torch.int32),
        cluster=torch.full(shape, -1, device=device, dtype=torch.int32),
        topic=torch.full(shape, -1, device=device, dtype=torch.int16),
        kind=torch.full(shape, -1, device=device, dtype=torch.int8),
        surface=torch.full(shape, -1, device=device, dtype=torch.int8),
        step=torch.full(shape, -1, device=device, dtype=torch.int32),
    )
    return ledger


def _initialize_user_seeds(
    config: TwinConfig, start: int, count: int, device,
) -> _UserSeeds:
    user_id = torch.arange(start, start + count, device=device, dtype=torch.long)
    long_interest = functional.softmax(
        normal(
            user_id, 0, 101, config.environment_seed, config.topics
        ), dim=1
    )
    lifecycle = torch.floor(
        uniform(user_id, 0, 139, config.seed) * 4
    ).long()
    country = torch.floor(
        uniform(user_id, 0, 149, config.seed) * config.countries
    ).long()
    activity_tier = torch.floor(
        uniform(user_id, 0, 197, config.seed) * 4
    ).long()
    socioeconomic = torch.floor(
        uniform(user_id, 0, 229, config.seed) * 5
    ).long()
    activity_rates = torch.tensor(
        [0.08, 0.24, 0.52, 0.82], device=device
    )
    initially_registered = (
        uniform(user_id, 0, 199, config.environment_seed)
        < config.initial_registered_fraction
    )
    horizon = config.preperiod_steps + config.measurement_steps * 10
    signup_step = 1 + torch.floor(
        uniform(user_id, 0, 211, config.environment_seed) * horizon
    ).long()
    signup_step = torch.where(
        initially_registered, torch.zeros_like(signup_step), signup_step
    )
    return _UserSeeds(
        user_id, long_interest, lifecycle, country, activity_tier,
        socioeconomic, activity_rates, signup_step, signup_step == 0,
    )


def _initialize_latent_users(
    config: TwinConfig, seeds: _UserSeeds, device,
) -> LatentUserState:
    user_id = seeds.user_id
    true_satisfaction = 0.35 + 0.45 * uniform(
        user_id, 0, 131, config.environment_seed
    )
    true_fatigue = 0.10 * uniform(
        user_id, 0, 137, config.environment_seed
    )
    true_conformity = uniform(
        user_id, 0, 257, config.environment_seed
    )
    true_spending = (
        0.10 + 0.16 * seeds.socioeconomic.float()
        + 0.14 * normal(user_id, 0, 233, config.environment_seed)
    ).clamp(0.02, 0.98)
    true_commerce = uniform(user_id, 0, 109, config.environment_seed)
    true_local = uniform(user_id, 0, 113, config.environment_seed)
    true_creator = uniform(user_id, 0, 127, config.environment_seed)
    true_surface = uniform(
        user_id, 0, 107, config.environment_seed, len(Surface)
    )
    return LatentUserState(
        long_interest=seeds.long_interest,
        short_interest=seeds.long_interest.clone(),
        satisfaction=true_satisfaction,
        fatigue=true_fatigue,
        conformity=true_conformity,
        spending_power=true_spending,
        commerce_intent=true_commerce,
        local_intent=true_local,
        creator_intent=true_creator,
        activity_propensity=(
            seeds.activity_rates[seeds.activity_tier]
            + 0.08 * normal(user_id, 0, 397, config.environment_seed)
        ).clamp(0.02, 0.98),
        surface_intent=true_surface,
        signup_step=seeds.signup_step,
        retained=torch.ones(
            len(user_id), device=device, dtype=torch.bool
        ),
        habit_strength=(
            0.12 + 0.72 * uniform(
                user_id, 0, 419, config.environment_seed
            )
        ),
    )


def _initialize_platform_users(
    config: TwinConfig, seeds: _UserSeeds, device,
) -> UserState:
    user_id = seeds.user_id
    count = len(user_id)
    observed_noise = normal(user_id, 0, 103, config.seed, config.topics)
    observed = functional.softmax(
        torch.log(seeds.long_interest.clamp_min(1e-8))
        + 0.35 * observed_noise,
        dim=1,
    )
    estimate_noise = 0.10 * normal(user_id, 0, 401, config.seed)
    return UserState(
        user_id=user_id,
        short_interest=observed.clone(),
        observed_interest=observed,
        surface_affinity_estimate=uniform(
            user_id, 0, 409, config.seed, len(Surface)
        ),
        commerce_intent_estimate=(
            0.18 + 0.10 * (seeds.activity_tier >= 2).float()
            + estimate_noise
        ).clamp(0.0, 1.0),
        local_intent_estimate=(
            0.20 + 0.08 * (seeds.activity_tier >= 1).float()
            - estimate_noise
        ).clamp(0.0, 1.0),
        creator_intent_estimate=(
            0.06 + 0.05 * (seeds.activity_tier >= 2).float()
            + 0.5 * estimate_noise
        ).clamp(0.0, 1.0),
        query_topic=torch.remainder(
            user_id * 48_271 + 17, config.topics
        ),
        query_strength=uniform(user_id, 0, 271, config.seed),
        satisfaction_estimate=torch.full(
            (count,), 0.50, device=device
        ),
        fatigue_counter=torch.zeros(count, device=device),
        lifecycle=seeds.lifecycle,
        country=seeds.country,
        region=(
            seeds.country * config.regions_per_country
            + torch.floor(
                uniform(user_id, 0, 227, config.seed)
                * config.regions_per_country
            ).long()
        ),
        timezone_offset=(seeds.country * 7 + 5).remainder(25) - 12,
        socioeconomic=seeds.socioeconomic,
        spending_power_estimate=(
            0.10 + 0.16 * seeds.socioeconomic.float() + estimate_noise
        ).clamp(0.0, 1.0),
        activity_tier=seeds.activity_tier,
        activity_rate_estimate=seeds.activity_rates[seeds.activity_tier],
        acquisition_channel=torch.floor(
            uniform(user_id, 0, 239, config.seed) * 5
        ).long(),
        signup_step=torch.where(
            seeds.registered, torch.zeros_like(seeds.signup_step),
            torch.full_like(seeds.signup_step, -1),
        ),
        tenure_days=torch.where(
            seeds.registered,
            torch.floor(1.0 + 1_200.0 * uniform(
                user_id, 0, 241, config.seed
            )).long(),
            torch.zeros_like(seeds.signup_step),
        ),
        cold_start_confidence=torch.where(
            seeds.registered,
            0.15 + 0.75 * uniform(user_id, 0, 251, config.seed),
            torch.zeros(count, device=device),
        ),
        trend_affinity_estimate=torch.full(
            (count,), 0.50, device=device
        ),
        registered=seeds.registered,
        active=seeds.registered & (
            uniform(user_id, 0, 223, config.seed)
            < seeds.activity_rates[seeds.activity_tier]
        ),
        session_depth=torch.zeros(count, device=device, dtype=torch.long),
        request_index=torch.zeros(count, device=device, dtype=torch.long),
        ledger=_initialize_ledger(config, count, device),
    )


def initialize_user_pair(config: TwinConfig, start: int, count: int, device):
    seeds = _initialize_user_seeds(config, start, count, device)
    return (
        _initialize_platform_users(config, seeds, device),
        _initialize_latent_users(config, seeds, device),
    )


def initialize_users(config: TwinConfig, start: int, count: int, device):
    return initialize_user_pair(config, start, count, device)[0]


def initialize_catalog_pair(config: TwinConfig, device):
    item_id = torch.arange(config.catalog_items, device=device, dtype=torch.long)
    kind = torch.remainder(item_id, len(ItemKind))
    topic = torch.remainder(item_id * 69_697 + 29, config.topics)
    basis = torch.eye(config.topics, device=device)[topic]
    noise = normal(
        item_id, 0, 151, config.environment_seed, config.topics
    )
    semantic_embedding = functional.normalize(basis + 0.18 * noise, dim=1)
    content_noise = normal(item_id, 0, 353, config.seed, config.topics)
    embedding = functional.normalize(
        semantic_embedding + 0.22 * content_noise, dim=1
    )
    true_quality = torch.sigmoid(normal(
        item_id, 0, 157, config.environment_seed
    ))
    quality = torch.sigmoid(
        torch.logit(true_quality.clamp(1e-5, 1.0 - 1e-5))
        + 0.55 * normal(item_id, 0, 359, config.seed)
    )
    true_risk = (
        0.7 * uniform(item_id, 0, 173, config.environment_seed)
        + 0.3 * (1.0 - true_quality)
    ).clamp(0.0, 1.0)
    risk = (
        true_risk + 0.15 * normal(item_id, 0, 367, config.seed)
    ).clamp(0.0, 1.0)
    popularity = uniform(item_id, 0, 163, config.seed).pow(3.0)
    catalog = CatalogState(
        item_id=item_id,
        kind=kind,
        topic=topic,
        topic_embedding=embedding,
        author=torch.remainder(item_id * 7_919 + 31, config.creators),
        cluster=torch.remainder(item_id * 1_009 + topic * 17, config.catalog_items),
        country=torch.remainder(item_id * 503 + 37, config.countries),
        region=torch.remainder(
            item_id * 1_009 + 41,
            config.countries * config.regions_per_country,
        ),
        quality=quality,
        text_quality=uniform(item_id, 0, 277, config.seed),
        visual_quality=uniform(item_id, 0, 281, config.seed),
        duration_seconds=(
            4.0 + 176.0 * uniform(item_id, 0, 283, config.seed).square()
        ),
        freshness=uniform(item_id, 0, 167, config.seed),
        popularity=popularity,
        risk=risk,
        price_match_prior=uniform(item_id, 0, 179, config.seed),
        price=torch.exp(
            -1.5 + 5.0 * uniform(item_id, 0, 293, config.seed)
        ),
        merchant_quality=uniform(item_id, 0, 307, config.seed),
        inventory=uniform(item_id, 0, 181, config.seed),
        sponsored_value=uniform(item_id, 0, 191, config.seed),
        ad_bid=(
            0.05 + 2.95 * uniform(item_id, 0, 311, config.seed)
        ),
        ad_budget=(
            10.0 + 990.0 * uniform(item_id, 0, 313, config.seed)
        ),
        ad_spend=torch.zeros(config.catalog_items, device=device),
        live_start_hour=torch.floor(
            24.0 * uniform(item_id, 0, 317, config.seed)
        ).long(),
        live_duration_hours=(
            1 + torch.floor(
                6.0 * uniform(item_id, 0, 331, config.seed)
            ).long()
        ),
        poi_open_hour=torch.floor(
            10.0 * uniform(item_id, 0, 337, config.seed)
        ).long(),
        poi_close_hour=(
            17 + torch.floor(
                7.0 * uniform(item_id, 0, 347, config.seed)
            ).long()
        ).clamp_max(23),
        supply_exposure=torch.zeros(config.catalog_items, device=device),
        supply_positive=torch.zeros(config.catalog_items, device=device),
        supply_negative=torch.zeros(config.catalog_items, device=device),
        supply_payment=torch.zeros(config.catalog_items, device=device),
        creator_motivation=(
            0.30 + 0.55 * uniform(
                torch.arange(config.creators, device=device),
                0, 193, config.seed,
            )
        ),
        creator_active=torch.ones(
            config.creators, device=device, dtype=torch.bool
        ),
        creator_posts=torch.bincount(
            torch.remainder(item_id * 7_919 + 31, config.creators),
            minlength=config.creators,
        ).float(),
    )
    latent = LatentCatalogState(
        semantic_embedding=semantic_embedding,
        true_quality=true_quality,
        true_risk=true_risk,
        price_appeal=uniform(
            item_id, 0, 373, config.environment_seed
        ),
    )
    return catalog, latent


def initialize_catalog(config: TwinConfig, device):
    return initialize_catalog_pair(config, device)[0]
