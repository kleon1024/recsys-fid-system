"""Cohort-aware arrival, return-survival, churn, and reactivation authority."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....randomness.counter import uniform
from ..state import HiddenUserState


CALENDAR_VERSION = "session-survival-calendar-v1"


@dataclass(frozen=True)
class ReturnOutcome:
    delay_ticks: torch.Tensor
    churned: torch.Tensor
    reactivation_time: torch.Tensor


def arrival_hazard(
    users: HiddenUserState,
    logical_time: int,
    ticks_per_day: int,
) -> torch.Tensor:
    local_hour = (
        logical_time * 24.0 / ticks_per_day
        + users.timezone_offset.float()
    ).remainder(24.0)
    day = torch.div(
        torch.full_like(users.user_id, logical_time),
        ticks_per_day,
        rounding_mode="floor",
    ).remainder(7)
    circadian = (
        0.14
        + 0.36 * torch.exp(-((local_hour - 8.0) / 2.8).square())
        + 0.48 * torch.exp(-((local_hour - 12.5) / 3.8).square())
        + 0.78 * torch.exp(-((local_hour - 21.0) / 3.2).square())
    )
    lifecycle_multiplier = torch.tensor(
        (1.25, 0.48, 0.92, 1.38),
        device=users.user_id.device,
    )[users.lifecycle_cohort.clamp(0, 3)]
    weekly = torch.gather(users.weekly_activity, 1, day[:, None]).squeeze(1)
    return (
        users.activity
        * circadian
        * weekly
        * lifecycle_multiplier
        * (0.38 + 0.62 * users.habit)
        * (0.48 + 0.52 * users.satisfaction)
        * (1.0 - 0.38 * users.fatigue)
        / ticks_per_day
    ).clamp(0.0, 0.55)


def sample_return_outcome(
    users: HiddenUserState,
    event_id: torch.Tensor,
    event_time: torch.Tensor,
    ticks_per_day: int,
    seed: int,
) -> ReturnOutcome:
    noise = uniform(event_id, 0, 1_503, seed).clamp(1e-6, 1.0 - 1e-6)
    scale_days = torch.exp(
        1.05
        + 1.45 * users.fatigue
        - 1.35 * users.satisfaction
        - 1.10 * users.habit
        + 0.70 * users.churn_susceptibility
    ).clamp(0.02, 60.0)
    shape = (0.72 + 0.90 * users.habit).clamp(0.55, 1.65)
    delay_days = scale_days * (-torch.log1p(-noise)).pow(1.0 / shape)
    delay_ticks = torch.ceil(delay_days * ticks_per_day).long().clamp_min(1)
    churn_probability = torch.sigmoid(
        -3.7
        + 2.7 * users.churn_susceptibility
        + 2.2 * users.fatigue
        - 2.4 * users.satisfaction
        - 1.2 * users.habit
        - 0.10 * torch.log1p(users.session_count.float())
    )
    churned = uniform(event_id, 0, 1_509, seed) < churn_probability
    can_reactivate = uniform(event_id, 0, 1_513, seed) < (
        0.08 + 0.22 * users.habit
    )
    reactivation_days = 21.0 + 160.0 * uniform(event_id, 0, 1_519, seed).square()
    reactivation = event_time + torch.ceil(
        reactivation_days * ticks_per_day
    ).long()
    never = torch.full_like(reactivation, torch.iinfo(torch.long).max // 4)
    reactivation = torch.where(churned & can_reactivate, reactivation, never)
    return ReturnOutcome(delay_ticks, churned, reactivation)
