"""Exogenous UG campaigns and endogenous product-led acquisition pressure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from ....randomness.counter import normal, uniform
from ...contracts import AppEventBatch, EventType


ACQUISITION_VERSION = "ug-plg-acquisition-v1"


class AcquisitionChannel(IntEnum):
    ORGANIC = 0
    PAID = 1
    REFERRAL = 2
    CREATOR_LED = 3
    CROSS_PRODUCT = 4


@dataclass(frozen=True)
class AcquisitionPopulation:
    channel: torch.Tensor
    quality: torch.Tensor
    referral_susceptibility: torch.Tensor


@dataclass
class GrowthState:
    campaign_intensity: torch.Tensor
    referral_pressure: torch.Tensor
    creator_pressure: torch.Tensor
    last_time: int


def sample_acquisition_population(
    user: torch.Tensor,
    country: torch.Tensor,
    segment: torch.Tensor,
    habit: torch.Tensor,
    seed: int,
) -> AcquisitionPopulation:
    channel = torch.arange(len(AcquisitionChannel), device=user.device).float()
    logits = torch.stack((
        0.70 + 0.18 * habit,
        0.10 + 0.12 * (segment == 1).float(),
        0.18 + 0.38 * habit,
        0.08 + 0.14 * (segment == 4).float(),
        -0.05 + 0.10 * country.remainder(3).float(),
    ), dim=1)
    logits += 0.08 * torch.sin(
        (country.float()[:, None] + 1.0) * (channel[None] + 1.0),
    )
    draw = uniform(
        user, 0, 1_611, seed, len(AcquisitionChannel),
    ).clamp(1e-6, 1.0 - 1e-6)
    selected = torch.argmax(logits - torch.log(-torch.log(draw)), dim=1)
    quality_prior = torch.tensor(
        (0.58, 0.42, 0.68, 0.63, 0.50), device=user.device,
    )[selected]
    quality = torch.sigmoid(
        torch.logit(quality_prior)
        + 0.45 * normal(user, 0, 1_613, seed)
        + 0.30 * (habit - 0.5)
    )
    referral = torch.sigmoid(
        -0.7
        + 1.4 * habit
        + 0.35 * normal(user, 0, 1_617, seed)
    )
    return AcquisitionPopulation(selected, quality, referral)


class GrowthProcess:
    """Own campaign seasonality and product-led referral externalities."""

    def __init__(self, countries: int, seed: int, device: torch.device):
        self.seed = seed
        self.state = GrowthState(
            campaign_intensity=torch.ones(countries, device=device),
            referral_pressure=torch.zeros(countries, device=device),
            creator_pressure=torch.zeros(countries, device=device),
            last_time=-1,
        )

    def advance(self, logical_time: int, ticks_per_day: int) -> None:
        if logical_time <= self.state.last_time:
            return
        country = torch.arange(
            len(self.state.campaign_intensity),
            device=self.state.campaign_intensity.device,
        )
        day = logical_time / ticks_per_day
        weekly = torch.sin(2.0 * torch.pi * day / 7.0 + 0.41 * country)
        campaign_noise = normal(country, logical_time, 1_619, self.seed)
        self.state.campaign_intensity.copy_(
            torch.exp(0.22 * weekly + 0.08 * campaign_noise).clamp(0.55, 1.85)
        )
        self.state.referral_pressure.mul_(0.985)
        self.state.creator_pressure.mul_(0.992)
        self.state.last_time = logical_time

    def registration_probability(
        self,
        channel: torch.Tensor,
        quality: torch.Tensor,
        referral_susceptibility: torch.Tensor,
        country: torch.Tensor,
    ) -> torch.Tensor:
        base = torch.tensor(
            (0.045, 0.070, 0.035, 0.030, 0.040),
            device=channel.device,
        )[channel]
        campaign = self.state.campaign_intensity[country]
        referral = self.state.referral_pressure[country]
        creator = self.state.creator_pressure[country]
        multiplier = torch.ones_like(base)
        multiplier = torch.where(
            channel == int(AcquisitionChannel.PAID), campaign, multiplier,
        )
        multiplier = torch.where(
            channel == int(AcquisitionChannel.REFERRAL),
            1.0 + referral_susceptibility * referral,
            multiplier,
        )
        multiplier = torch.where(
            channel == int(AcquisitionChannel.CREATOR_LED),
            1.0 + creator,
            multiplier,
        )
        return (base * multiplier * (0.55 + 0.9 * quality)).clamp(0.0, 0.65)

    def commit(self, events: AppEventBatch) -> None:
        valid = (events.user_id >= 0) & (events.country >= 0)
        referral = valid & (
            events.event(EventType.SHARE)
            | events.event(EventType.FOLLOW)
        )
        creator = valid & events.event(EventType.PUBLISH)
        self._accumulate(self.state.referral_pressure, events.country[referral], 0.003)
        self._accumulate(self.state.creator_pressure, events.country[creator], 0.006)

    @staticmethod
    def _accumulate(target: torch.Tensor, country: torch.Tensor, weight: float) -> None:
        if not len(country):
            return
        increment = torch.zeros_like(target)
        increment.index_add_(0, country, torch.full_like(country, weight, dtype=torch.float))
        target.add_(increment).clamp_(0.0, 3.0)

