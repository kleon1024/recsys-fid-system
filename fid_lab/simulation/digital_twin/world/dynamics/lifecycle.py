"""Sufficient-state transitions for acquisition, activation, and retention."""

from __future__ import annotations

import torch

from ....randomness.counter import uniform
from .needs import refresh_expired_needs


LIFECYCLE_DYNAMICS_VERSION = "activation-retention-state-v2-q1e5"
STATE_QUANTIZATION_SCALE = 100_000.0


def advance_latent_user_state(state, growth, logical_time: int, config):
    user = state.user_id
    refresh_expired_needs(
        user=user,
        need_kind=state.need_kind,
        need_topic=state.need_topic,
        need_strength=state.need_strength,
        need_expiry_time=state.need_expiry_time,
        primary_topic=state.primary_topic,
        secondary_topic=state.secondary_topic,
        logical_time=logical_time,
        ticks_per_day=config.ticks_per_day,
        topics=config.topics,
        seed=config.environment_seed,
    )
    candidate = (~state.registered) & (state.signup_time <= logical_time)
    if config.initialization_mode == "bootstrap":
        return candidate
    probability = growth.registration_probability(
        state.acquisition_channel,
        state.acquisition_quality,
        state.referral_susceptibility,
        state.country,
    )
    return candidate & (
        uniform(user, logical_time, 1_633, config.environment_seed) < probability
    )


def commit_session_start(
    state,
    user: torch.Tensor,
    event_time: torch.Tensor,
    ticks_per_day: int,
) -> None:
    if not len(user):
        return
    start_day = torch.div(event_time, ticks_per_day, rounding_mode="floor")
    previous_day = state.last_active_day[user]
    returned_next_day = (previous_day >= 0) & (start_day - previous_day <= 2)
    state.return_streak[user] = torch.where(
        returned_next_day,
        state.return_streak[user] + 1,
        torch.ones_like(user),
    )
    state.last_active_time[user] = event_time
    state.last_active_day[user] = start_day


def update_lifecycle_stage(state, user: torch.Tensor) -> None:
    if not len(user):
        return
    stage = torch.ones_like(user)
    activated = (
        (state.activation_score[user] >= 0.52)
        & (state.session_count[user] >= 2)
    )
    retained = (
        (state.return_streak[user] >= 3)
        | (state.session_count[user] >= 10)
    )
    stage = torch.where(activated, torch.full_like(stage, 2), stage)
    stage = torch.where(retained, torch.full_like(stage, 3), stage)
    stage = torch.where(state.churned[user], torch.full_like(stage, 4), stage)
    state.lifecycle_stage[user] = stage


def commit_need_and_activation(
    state,
    touched: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    dwell: torch.Tensor,
    disappointment: torch.Tensor,
    repeat_pressure: torch.Tensor,
) -> None:
    value = (
        0.45 * positive.clamp_max(4.0)
        + 0.08 * dwell.clamp_max(10.0)
        - 0.80 * negative.clamp_max(2.0)
        - 0.55 * disappointment
        - 0.70 * repeat_pressure
    ).clamp(-2.0, 3.0)
    state.session_value_ema[touched] = (
        0.88 * state.session_value_ema[touched] + 0.12 * value[touched]
    )
    activation_signal = torch.sigmoid(
        value + torch.logit(state.acquisition_quality.clamp(0.01, 0.99))
    )
    state.activation_score[touched] = (
        0.94 * state.activation_score[touched]
        + 0.06 * activation_signal[touched]
    ).clamp(0.0, 1.0)
    next_need = (
        state.need_strength
        - 0.035 * positive.clamp_max(3.0)
        - 0.010 * dwell.clamp_max(8.0)
        + 0.070 * negative.clamp_max(2.0)
        + 0.055 * disappointment
    ).clamp(0.0, 1.0)
    state.need_strength[touched] = next_need[touched]
    update_lifecycle_stage(state, torch.where(touched)[0])


def quantize_dynamic_state(state) -> None:
    """Remove sub-resolution CUDA reduction noise from checkpoint state."""
    for name in (
        "short_interest",
        "satisfaction",
        "fatigue",
        "habit",
        "disappointment",
        "need_strength",
        "activation_score",
        "session_value_ema",
    ):
        value = getattr(state, name)
        value.mul_(STATE_QUANTIZATION_SCALE).round_().div_(
            STATE_QUANTIZATION_SCALE,
        )
