"""Versioned correlated latent-mixture population authority."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....randomness.counter import normal, uniform
from ...contracts import Surface


POPULATION_VERSION = "mau-conditioned-correlated-latent-mixture-v3"


@dataclass(frozen=True)
class PopulationSample:
    mixture: torch.Tensor
    factors: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    timezone_offset: torch.Tensor
    language: torch.Tensor
    device_class: torch.Tensor
    lifecycle_cohort: torch.Tensor
    weekly_activity: torch.Tensor
    churn_susceptibility: torch.Tensor
    primary_topic: torch.Tensor
    secondary_topic: torch.Tensor
    satisfaction: torch.Tensor
    fatigue: torch.Tensor
    habit: torch.Tensor
    activity: torch.Tensor
    novelty: torch.Tensor
    spending_power: torch.Tensor
    surface_intent: torch.Tensor
    response_style: torch.Tensor


def _constant(values, device) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=torch.float)


def _mixture(user: torch.Tensor, seed: int) -> torch.Tensor:
    weights = _constant((0.24, 0.22, 0.18, 0.16, 0.12, 0.08), user.device)
    draw = uniform(user, 0, 1_401, seed)
    return torch.searchsorted(weights.cumsum(0), draw).clamp_max(5)


def _correlated_factors(
    user: torch.Tensor, mixture: torch.Tensor, seed: int,
) -> torch.Tensor:
    location = _constant((
        (0.55, -0.35, 0.70, -0.30, -0.20, 0.10),
        (-0.45, 0.35, -0.55, 0.60, -0.10, -0.20),
        (0.10, -0.20, 0.20, 0.55, 0.35, -0.25),
        (-0.15, 0.65, -0.25, -0.15, 0.10, 0.40),
        (0.35, -0.10, 0.05, 0.15, 0.85, 0.20),
        (-0.55, 0.25, -0.20, -0.05, -0.35, 0.90),
    ), user.device)
    raw = normal(user, 0, 1_409, seed, 6)
    shared = raw[:, :3]
    factors = torch.stack((
        raw[:, 0],
        0.58 * raw[:, 0] + 0.82 * raw[:, 1],
        0.50 * raw[:, 0] - 0.35 * raw[:, 1] + 0.79 * raw[:, 2],
        0.42 * shared.sum(1) + 0.68 * raw[:, 3],
        -0.30 * raw[:, 0] + 0.25 * raw[:, 2] + 0.88 * raw[:, 4],
        0.25 * raw[:, 1] + 0.30 * raw[:, 3] + 0.86 * raw[:, 5],
    ), dim=1)
    return factors + location[mixture]


def _traits(
    user: torch.Tensor, mixture: torch.Tensor, factors: torch.Tensor, seed: int,
) -> torch.Tensor:
    loadings = _constant((
        (0.78, -0.22, 0.42, 0.00, 0.00, 0.18),
        (-0.34, 0.86, -0.18, 0.28, 0.00, 0.00),
        (0.38, -0.12, 0.88, -0.30, 0.00, 0.00),
        (0.52, 0.08, 0.72, 0.38, 0.00, 0.00),
        (-0.18, 0.12, -0.48, 0.82, 0.24, 0.00),
        (0.08, 0.00, 0.00, 0.18, 0.92, 0.12),
    ), user.device)
    intercept = _constant((
        (0.20, -1.15, 0.25, -0.45, -0.20, -0.25),
        (-0.25, -0.75, -0.35, -1.00, 0.45, -0.55),
        (0.05, -0.95, 0.00, -0.65, 0.30, -0.20),
        (-0.10, -0.35, -0.20, -0.55, -0.05, 0.15),
        (0.30, -1.10, 0.10, -0.40, 0.00, 0.75),
        (-0.35, -0.55, -0.15, -1.10, -0.25, 0.35),
    ), user.device)
    residual = 0.22 * normal(user, 0, 1_421, seed, 6)
    return torch.sigmoid(intercept[mixture] + factors @ loadings.T + residual)


def _geography(
    user: torch.Tensor,
    mixture: torch.Tensor,
    factors: torch.Tensor,
    countries: int,
    regions_per_country: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    country_index = torch.arange(countries, device=user.device).float()
    phase = 2.0 * torch.pi * country_index / max(countries, 1)
    logits = (
        0.75 * factors[:, 0, None] * torch.cos(phase)[None]
        + 0.55 * factors[:, 4, None] * torch.sin(phase)[None]
        + 0.28 * torch.cos(phase[None] - mixture[:, None].float())
    )
    draw = uniform(user, 0, 1_433, seed, countries).clamp(1e-6, 1.0 - 1e-6)
    country = torch.argmax(logits - torch.log(-torch.log(draw)), dim=1)
    region_draw = uniform(user, 0, 1_439, seed)
    local_region = torch.floor(region_draw * regions_per_country).long().clamp_max(
        regions_per_country - 1
    )
    region = country * regions_per_country + local_region
    base_timezone = torch.round(
        -11.0 + 22.0 * country.float() / max(countries - 1, 1)
    ).long()
    timezone = (base_timezone + local_region.remainder(3) - 1).clamp(-12, 14)
    language = torch.remainder(country * 3 + mixture, max(4, countries))
    return country, region, timezone, language


def _topics(
    user: torch.Tensor,
    mixture: torch.Tensor,
    factors: torch.Tensor,
    country: torch.Tensor,
    topics: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topic = torch.arange(topics, device=user.device).float()
    phase = 2.0 * torch.pi * topic / max(topics, 1)
    affinity = (
        factors[:, 0, None] * torch.sin(phase)[None]
        + factors[:, 2, None] * torch.cos(phase)[None]
        + 0.35 * torch.cos(
            phase[None]
            - 0.47 * mixture[:, None].float()
            - 0.19 * country[:, None].float()
        )
    )
    first_draw = uniform(user, 0, 1_447, seed, topics).clamp(1e-6, 1.0 - 1e-6)
    primary = torch.argmax(affinity - torch.log(-torch.log(first_draw)), dim=1)
    second_draw = uniform(user, 0, 1_449, seed, topics).clamp(1e-6, 1.0 - 1e-6)
    second_score = affinity - torch.log(-torch.log(second_draw))
    second_score.scatter_(1, primary[:, None], -torch.inf)
    return primary, torch.argmax(second_score, dim=1)


def _surface_and_style(
    user: torch.Tensor, factors: torch.Tensor, traits: torch.Tensor, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    surface_loadings = _constant((
        (0.55, -0.20, 0.60, 0.10, -0.15, -0.10),
        (0.05, -0.05, -0.20, 0.55, 0.10, -0.10),
        (-0.05, 0.00, -0.10, 0.15, 0.70, 0.10),
        (0.10, 0.25, 0.10, 0.30, -0.05, 0.20),
        (0.00, -0.05, 0.00, 0.20, 0.35, 0.15),
        (-0.10, -0.10, 0.35, 0.25, 0.00, 0.10),
    ), user.device)
    logits = factors @ surface_loadings.T
    logits[:, int(Surface.FEED)] += 1.35 + 0.45 * traits[:, 2]
    logits[:, int(Surface.SEARCH)] += 0.20 + 0.35 * traits[:, 4]
    logits[:, int(Surface.POSTING)] += 0.30 * traits[:, 2]
    surface_intent = torch.softmax(logits, dim=1)
    row = torch.arange(1, 9, device=user.device).float()[:, None]
    column = torch.arange(1, 7, device=user.device).float()[None]
    loadings = 0.48 * torch.sin(row * column * 0.73)
    style = factors @ loadings.T + 0.28 * normal(user, 0, 1_457, seed, 8)
    return surface_intent, style.clamp(-3.0, 3.0)


def _weekly_and_churn(
    user: torch.Tensor, mixture: torch.Tensor, factors: torch.Tensor, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    day = torch.arange(7, device=user.device).float()
    phase = 2.0 * torch.pi * day / 7.0
    weekly_log = (
        0.28 * factors[:, 1, None] * torch.sin(phase)[None]
        + 0.24 * factors[:, 3, None] * torch.cos(phase)[None]
    )
    weekend = (day >= 5).float()[None]
    weekly_log += weekend * (0.12 + 0.08 * mixture[:, None].float())
    weekly = torch.exp(weekly_log.clamp(-1.2, 1.2))
    weekly /= weekly.mean(dim=1, keepdim=True)
    churn = torch.sigmoid(
        -0.45
        + 0.62 * factors[:, 1]
        - 0.52 * factors[:, 2]
        + 0.20 * factors[:, 3]
        + 0.18 * normal(user, 0, 1_463, seed)
    )
    return weekly, churn


def sample_population(
    user: torch.Tensor,
    *,
    topics: int,
    countries: int,
    regions_per_country: int,
    seed: int,
) -> PopulationSample:
    """Sample a correlated ecosystem population with request-order invariance."""
    mixture = _mixture(user, seed)
    factors = _correlated_factors(user, mixture, seed)
    traits = _traits(user, mixture, factors, seed)
    country, region, timezone, language = _geography(
        user, mixture, factors, countries, regions_per_country, seed,
    )
    primary, secondary = _topics(
        user, mixture, factors, country, topics, seed,
    )
    surface_intent, response_style = _surface_and_style(
        user, factors, traits, seed,
    )
    weekly_activity, churn_susceptibility = _weekly_and_churn(
        user, mixture, factors, seed,
    )
    # The simulation population represents rolling-28-day active users rather
    # than all historical registrations. Preserve heterogeneous frequency
    # without admitting a lifetime-account-sized dormant mass.
    activity = 0.08 + 0.82 * traits[:, 3]
    lifecycle = torch.where(
        activity > 0.42,
        torch.full_like(mixture, 3),
        torch.where(
            activity > 0.16,
            torch.full_like(mixture, 2),
            torch.ones_like(mixture),
        ),
    )
    device_score = traits[:, 5] + 0.18 * factors[:, 3]
    device_class = torch.bucketize(
        device_score, _constant((0.34, 0.66), user.device)
    )
    return PopulationSample(
        mixture=mixture,
        factors=factors,
        country=country,
        region=region,
        timezone_offset=timezone,
        language=language,
        device_class=device_class,
        lifecycle_cohort=lifecycle,
        weekly_activity=weekly_activity,
        churn_susceptibility=churn_susceptibility,
        primary_topic=primary,
        secondary_topic=secondary,
        satisfaction=traits[:, 0],
        fatigue=traits[:, 1],
        habit=traits[:, 2],
        activity=activity,
        novelty=traits[:, 4],
        spending_power=traits[:, 5],
        surface_intent=surface_intent,
        response_style=response_style,
    )
