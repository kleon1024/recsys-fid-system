"""Hidden acquisition, app sessions, churn, and reactivation lifecycle."""

from __future__ import annotations

import torch

from ...randomness.counter import uniform
from ..contracts import TwinConfig
from .latent import LatentUserState
from ..platform.state import UserState
from ..world.context import ContextState


def start_step_population(
    config: TwinConfig,
    users: UserState,
    latent_users: LatentUserState,
    context: ContextState,
    step: int,
) -> dict[str, int]:
    new_user = (
        (~users.registered) & latent_users.retained
        & (latent_users.signup_step <= step)
    )
    users.registered |= new_user
    users.signup_step[new_user] = step
    users.tenure_days[new_user] = 0
    users.cold_start_confidence[new_user] = 0.05
    country_prior = context.country_topic_heat[users.country]
    users.observed_interest[new_user] = torch.nn.functional.normalize(
        0.75 * country_prior[new_user]
        + 0.25 * users.observed_interest[new_user], dim=1,
    )
    local_hour = torch.remainder(step + users.timezone_offset, 24).float()
    circadian = (
        0.35 + 0.35 * torch.exp(-((local_hour - 12.0) / 5.0).square())
        + 0.30 * torch.exp(-((local_hour - 21.0) / 4.0).square())
    )
    arrival_probability = (
        latent_users.activity_propensity * circadian
        * context.region_activity[users.region]
        * (0.55 + 0.45 * latent_users.satisfaction)
    ).clamp(0.0, 0.95)
    arrived = users.registered & latent_users.retained & (~users.active) & (
        uniform(
            users.user_id, step, 431, config.environment_seed
        ) < arrival_probability
    )
    users.active |= arrived | new_user
    users.session_depth[arrived | new_user] = 0
    query_probability = (
        0.03 + 0.18 * latent_users.local_intent
        + 0.20 * latent_users.commerce_intent
    ).clamp_max(0.45)
    query_event = users.active & (
        uniform(
            users.user_id, step, 433, config.environment_seed
        ) < query_probability
    )
    query_interest = (
        users.cold_start_confidence[:, None] * latent_users.short_interest
        + (1.0 - users.cold_start_confidence[:, None]) * country_prior
    )
    users.query_topic = torch.where(
        query_event, query_interest.argmax(dim=1), users.query_topic
    )
    users.query_strength = torch.where(
        query_event,
        0.55 + 0.45 * uniform(
            users.user_id, step, 439, config.environment_seed
        ),
        0.88 * users.query_strength,
    )
    return {
        "new_users": int(new_user.sum()),
        "session_arrivals": int((arrived | new_user).sum()),
        "registered_users": int(users.registered.sum()),
    }
