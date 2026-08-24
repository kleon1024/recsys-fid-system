"""Environment facade that owns all hidden user-world transitions."""

from __future__ import annotations

import torch

from ...randomness.counter import uniform
from ..contracts import TwinConfig
from ..exchange import ObservableResponse, ServedSlate
from ..platform.state import CatalogState, UserState
from ..world.context import ContextState
from .latent import LatentCatalogState, LatentUserState
from .lifecycle import start_step_population
from .response import LatentBehaviorWorld, advance_hidden_state
from .traffic import route_surface


class UserEnvironment:
    """Private app-user world; the platform receives only emitted events."""

    def __init__(self, config: TwinConfig, device):
        self.config = config
        self.response_model = LatentBehaviorWorld(config, device)

    def begin_step(
        self,
        users: UserState,
        latent_users: LatentUserState,
        context: ContextState,
        step: int,
    ) -> torch.Tensor:
        start_step_population(
            self.config, users, latent_users, context, step
        )
        return route_surface(
            users, latent_users, context, step,
            self.config.environment_seed
        )

    def respond(
        self,
        users: UserState,
        latent_users: LatentUserState,
        catalog: CatalogState,
        latent_catalog: LatentCatalogState,
        context: ContextState,
        slate: ServedSlate,
        surface: torch.Tensor,
        step: int,
    ) -> ObservableResponse:
        return self.response_model.sample(
            users, latent_users, catalog, latent_catalog, context,
            slate, surface, step,
        )

    def commit(
        self,
        users: UserState,
        latent_users: LatentUserState,
        latent_catalog: LatentCatalogState,
        response: ObservableResponse,
        surface: torch.Tensor,
        step: int,
    ) -> None:
        advance_hidden_state(
            users, latent_users, latent_catalog, response,
            surface, step, self.config.environment_seed,
        )

    def advance_day(
        self,
        users: UserState,
        latent_users: LatentUserState,
        day: int,
    ) -> dict[str, int]:
        return_probability = torch.sigmoid(
            -0.4 + 1.7 * latent_users.satisfaction
            - 1.0 * latent_users.fatigue
            + 0.18 * users.lifecycle.float()
            + 0.55 * latent_users.habit_strength
        )
        churn_probability = torch.sigmoid(
            -5.2 - 1.4 * latent_users.satisfaction
            + 2.3 * latent_users.fatigue
            - 1.1 * latent_users.habit_strength
        )
        churned = users.registered & latent_users.retained & (
            uniform(users.user_id, day, 349, self.config.environment_seed)
            < churn_probability
        )
        latent_users.retained &= ~churned
        users.active &= latent_users.retained
        reacquired = users.registered & (~latent_users.retained) & (
            uniform(users.user_id, day, 353, self.config.environment_seed)
            < (0.001 + 0.006 * latent_users.habit_strength)
        )
        latent_users.retained |= reacquired
        returned = (~users.active) & latent_users.retained & (
            uniform(users.user_id, day, 347, self.config.environment_seed)
            < return_probability
        )
        users.active |= returned
        users.session_depth.zero_()
        users.fatigue_counter.mul_(0.70)
        latent_users.fatigue.mul_(0.70)
        latent_users.habit_strength.add_(
            0.015 * latent_users.satisfaction
            - 0.012 * latent_users.fatigue
        ).clamp_(0.02, 0.98)
        return {
            "churned_users": int(churned.sum()),
            "reacquired_users": int(reacquired.sum()),
            "returned_users": int(returned.sum()),
        }
