"""Hidden arrival, surface-entry, and request-context mechanism."""

from __future__ import annotations

import torch

from ...randomness.counter import uniform
from ..contracts import Surface
from ..platform.state import UserState
from ..world.context import ContextState
from .latent import LatentUserState


def route_surface(
    users: UserState,
    latent_users: LatentUserState,
    context: ContextState,
    step: int,
    seed: int,
) -> torch.Tensor:
    depth = users.session_depth.float()
    local_hour = torch.remainder(step + users.timezone_offset, 24)
    evening = (local_hour >= 18).float()
    workday = ((local_hour >= 8) & (local_hour <= 18)).float()
    region_activity = context.region_activity[users.region]
    intent = latent_users.surface_intent
    scores = torch.stack((
        1.9 + 0.35 * latent_users.satisfaction - 0.08 * depth
        + 0.20 * evening + 0.15 * region_activity,
        -0.7 + 1.6 * intent[:, Surface.SEARCH]
        + 0.35 * latent_users.local_intent
        + 0.30 * latent_users.commerce_intent + 0.18 * workday,
        -1.0 + 1.7 * latent_users.commerce_intent
        + 0.45 * intent[:, Surface.COMMERCE]
        + 0.25 * latent_users.spending_power,
        -1.1 + 1.3 * intent[:, Surface.LIVE] + 0.35 * evening,
        -0.9 + 1.7 * latent_users.local_intent
        + 0.40 * intent[:, Surface.LOCAL] + 0.15 * region_activity,
        -1.8 + 2.0 * latent_users.creator_intent
        + 0.30 * intent[:, Surface.POSTING] + 0.10 * evening,
    ), dim=1)
    probability = torch.softmax(scores, dim=1)
    draw = uniform(users.user_id, step, 211, seed)
    return (draw[:, None] > probability.cumsum(dim=1)).sum(dim=1).clamp_max(
        len(Surface) - 1
    )
