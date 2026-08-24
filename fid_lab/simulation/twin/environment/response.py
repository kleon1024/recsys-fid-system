"""Hidden learned-style multi-behavior response and state transition model."""

from __future__ import annotations

from math import sqrt

import torch

from ...randomness.counter import normal, uniform, uniform_for_items
from ..contracts import ItemKind, Surface, TwinConfig
from .latent import LatentCatalogState, LatentUserState
from ..exchange import (
    ObservableResponse,
    ServedSlate,
    TASK_INDEX,
    TASKS,
    task_applicability,
)
from ..platform.state import CatalogState, UserState
from ..world.context import ContextState

ResponseBatch = ObservableResponse


class LatentBehaviorWorld:
    """Fixed neural SCM; latent traits and weights are hidden from policies."""

    def __init__(self, config: TwinConfig, device):
        self.config = config
        self.device = torch.device(device)
        generator = torch.Generator(device=self.device).manual_seed(
            config.environment_seed + 300_001
        )
        inputs, width, experts = 34, 64, 5
        self.input_weight = torch.randn(
            inputs, width, generator=generator, device=self.device
        ) / sqrt(inputs)
        self.input_bias = 0.10 * torch.randn(
            width, generator=generator, device=self.device
        )
        self.expert_weight = 0.55 * torch.randn(
            experts, width, len(TASKS) + 1,
            generator=generator, device=self.device,
        ) / sqrt(width)
        self.expert_bias = 0.05 * torch.randn(
            experts, len(TASKS) + 1,
            generator=generator, device=self.device,
        )
        self.gate_weight = 0.40 * torch.randn(
            inputs, experts, generator=generator, device=self.device
        ) / sqrt(inputs)
        self.base_logit = torch.tensor(
            [
                1.2, 0.35, -0.45, -0.9, -2.0, -3.0, -3.2, -3.1,
                -1.4, -2.0, -3.0, -3.2, -4.0, -4.6, -3.4, -1.7, -2.6,
            ],
            device=self.device,
        )

    def _inputs(
        self, users: UserState, latent_users: LatentUserState,
        catalog: CatalogState, latent_catalog: LatentCatalogState,
        context: ContextState,
        selected_item: torch.Tensor, surface: torch.Tensor, step: int,
    ):
        item = selected_item.clamp_min(0)
        embedding = latent_catalog.semantic_embedding[item]
        long_affinity = (embedding * latent_users.long_interest).sum(dim=1)
        short_affinity = (embedding * latent_users.short_interest).sum(dim=1)
        author_fatigue = (
            users.ledger.author == catalog.author[item, None]
        ).sum(dim=1).float()
        cluster_fatigue = (
            users.ledger.cluster == catalog.cluster[item, None]
        ).sum(dim=1).float()
        topic_fatigue = (
            users.ledger.topic == catalog.topic[item, None]
        ).sum(dim=1).float()
        kind = catalog.kind[item]
        topic = catalog.topic[item]
        trend = (
            latent_users.conformity
            * context.country_topic_heat[users.country, topic]
            + (1.0 - latent_users.conformity)
            * context.global_topic_heat[topic]
        )
        local_hour = torch.remainder(step + users.timezone_offset, 24).float()
        affordability = torch.exp(-(
            torch.log1p(catalog.price[item])
            - latent_users.spending_power * 3.5
        ).abs())
        ad_pacing = (
            (catalog.ad_budget[item] - catalog.ad_spend[item]).clamp_min(0.0)
            / catalog.ad_budget[item]
        )
        live_elapsed = torch.remainder(
            step - catalog.live_start_hour[item], 24
        )
        available = torch.where(
            kind == int(ItemKind.LIVE_ROOM),
            (live_elapsed < catalog.live_duration_hours[item]).float(),
            torch.where(
                kind == int(ItemKind.POI),
                (
                    (local_hour >= catalog.poi_open_hour[item])
                    & (local_hour <= catalog.poi_close_hour[item])
                ).float(),
                catalog.inventory[item],
            ),
        )
        return torch.stack((
            long_affinity,
            short_affinity,
            latent_catalog.true_quality[item],
            catalog.freshness[item],
            catalog.popularity[item],
            latent_catalog.true_risk[item],
            latent_catalog.price_appeal[item],
            catalog.inventory[item],
            (catalog.country[item] == users.country).float(),
            latent_users.commerce_intent,
            latent_users.local_intent,
            latent_users.creator_intent,
            latent_users.satisfaction,
            latent_users.fatigue,
            users.lifecycle.float() / 3.0,
            surface.float() / max(len(Surface) - 1, 1),
            kind.float() / max(len(ItemKind) - 1, 1),
            author_fatigue.clamp_max(4.0) / 4.0,
            cluster_fatigue.clamp_max(4.0) / 4.0,
            topic_fatigue.clamp_max(4.0) / 4.0,
            trend,
            users.socioeconomic.float() / 4.0,
            latent_users.spending_power,
            users.activity_tier.float() / 3.0,
            users.cold_start_confidence,
            latent_users.conformity,
            torch.sin(2.0 * torch.pi * local_hour / 24.0),
            torch.cos(2.0 * torch.pi * local_hour / 24.0),
            (catalog.topic[item] == users.query_topic).float()
            * users.query_strength,
            (catalog.region[item] == users.region).float(),
            affordability,
            catalog.merchant_quality[item],
            ad_pacing,
            available,
        ), dim=1)

    def _choose_item(
        self,
        users: UserState,
        latent_users: LatentUserState,
        catalog: CatalogState,
        latent_catalog: LatentCatalogState,
        slate: ServedSlate,
        step: int,
    ) -> torch.Tensor:
        """Choose from the served slate using hidden utility, not rank score."""
        items = slate.exposed_item_ids
        valid = items >= 0
        safe_items = items.clamp_min(0)
        embedding = latent_catalog.semantic_embedding[safe_items]
        long_affinity = torch.einsum(
            "bkd,bd->bk", embedding, latent_users.long_interest
        )
        short_affinity = torch.einsum(
            "bkd,bd->bk", embedding, latent_users.short_interest
        )
        positions = torch.arange(items.shape[1], device=items.device).float()
        position_bias = -0.32 * torch.log1p(positions)[None]
        noise_u = uniform_for_items(
            users.user_id, safe_items, step, 293,
            self.config.environment_seed
        ).clamp(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(noise_u))
        utility = (
            0.85 * long_affinity
            + 0.70 * short_affinity
            + 0.30 * latent_catalog.true_quality[safe_items]
            - 0.25 * latent_catalog.true_risk[safe_items]
            + position_bias
            + 0.20 * gumbel
        ).masked_fill(~valid, -1e9)
        selected_position = utility.argmax(dim=1)
        return items.gather(1, selected_position[:, None]).squeeze(1)

    @torch.inference_mode()
    def sample(
        self, users: UserState, latent_users: LatentUserState,
        catalog: CatalogState, latent_catalog: LatentCatalogState,
        context: ContextState,
        slate: ServedSlate, surface: torch.Tensor, step: int,
    ) -> ResponseBatch:
        item = self._choose_item(
            users, latent_users, catalog, latent_catalog, slate, step
        ).clamp_min(0)
        inputs = self._inputs(
            users, latent_users, catalog, latent_catalog, context,
            item, surface, step
        )
        hidden = torch.nn.functional.silu(
            inputs @ self.input_weight + self.input_bias
        )
        hidden = hidden + 0.25 * torch.sin(
            hidden * torch.roll(hidden, 7, dims=1)
        )
        expert = torch.einsum("bd,edk->bek", hidden, self.expert_weight)
        expert = expert + self.expert_bias[None]
        gate = torch.softmax(inputs @ self.gate_weight, dim=1)
        output = torch.einsum("be,bek->bk", gate, expert)
        probability = torch.sigmoid(output[:, : len(TASKS)] + self.base_logit)
        probability = torch.round(probability * 100_000.0) / 100_000.0
        draws = uniform_for_items(
            users.user_id, item[:, None].expand(-1, len(TASKS)),
            step, 307, self.config.environment_seed,
        )
        task = draws < probability
        kind = catalog.kind[item]
        mask = task_applicability(surface, kind)
        task &= mask & users.active[:, None]
        play = task[:, TASK_INDEX["play"]]
        task[:, TASK_INDEX["play_3s"]] &= play
        task[:, TASK_INDEX["long_view"]] &= task[:, TASK_INDEX["play_3s"]]
        task[:, TASK_INDEX["complete"]] &= task[:, TASK_INDEX["long_view"]]
        for name in ("like", "comment", "share", "follow"):
            task[:, TASK_INDEX[name]] &= play
        click = task[:, TASK_INDEX["click"]]
        task[:, TASK_INDEX["detail"]] &= click
        task[:, TASK_INDEX["favorite"]] &= task[:, TASK_INDEX["detail"]]
        task[:, TASK_INDEX["add_cart"]] &= click
        task[:, TASK_INDEX["order"]] &= click
        task[:, TASK_INDEX["payment"]] &= task[:, TASK_INDEX["order"]]
        task[:, TASK_INDEX["publish"]] &= task[:, TASK_INDEX["create"]]
        stay_latent = torch.sigmoid(output[:, -1])
        residual = normal(
            users.user_id, step, 313, self.config.environment_seed
        )
        duration = 8.0 + 82.0 * latent_catalog.true_quality[item]
        stay = (
            duration * (0.20 + 0.80 * stay_latent)
            * torch.exp(0.18 * residual.clamp(-2.0, 2.0))
            * (play | click | task[:, TASK_INDEX["create"]]).float()
        ).clamp(0.0, 180.0)
        return ResponseBatch(task, mask, stay, item, users.active.clone())


def advance_hidden_state(
    users: UserState,
    latent_users: LatentUserState,
    latent_catalog: LatentCatalogState,
    response: ResponseBatch,
    surface: torch.Tensor,
    step: int,
    seed: int,
) -> None:
    item = response.selected_item
    positive = (
        response.event("long_view") | response.event("like")
        | response.event("click") | response.event("order")
        | response.event("publish")
    )
    negative = response.event("negative")
    latent_embedding = latent_catalog.semantic_embedding[item]
    rate = (0.03 + 0.12 * positive.float())[:, None]
    latent_users.short_interest = torch.nn.functional.normalize(
        (1.0 - rate) * latent_users.short_interest
        + rate * latent_embedding, dim=1
    )
    satisfaction_delta = (
        0.012 * torch.log1p(response.stay_seconds)
        + 0.04 * positive.float() - 0.12 * negative.float()
    )
    next_satisfaction = (
        0.985 * latent_users.satisfaction + satisfaction_delta
    ).clamp(0.0, 1.0)
    latent_users.satisfaction = torch.where(
        response.active, next_satisfaction, latent_users.satisfaction
    )
    next_fatigue = (
        0.92 * latent_users.fatigue
        + 0.025 + 0.05 * negative.float()
        + 0.01 * users.session_depth.float()
    ).clamp(0.0, 1.0)
    latent_users.fatigue = torch.where(
        response.active, next_fatigue, latent_users.fatigue
    )
    latent_users.commerce_intent = (
        0.95 * latent_users.commerce_intent
        + 0.10 * response.event("add_cart").float()
        + 0.15 * response.event("order").float()
    ).clamp(0.0, 1.0)
    latent_users.local_intent = (
        0.96 * latent_users.local_intent
        + 0.10 * ((surface == int(Surface.LOCAL)) & positive).float()
    ).clamp(0.0, 1.0)
    latent_users.creator_intent = (
        0.97 * latent_users.creator_intent
        + 0.18 * response.event("publish").float()
    ).clamp(0.0, 1.0)
    latent_users.activity_propensity = (
        0.995 * latent_users.activity_propensity
        + 0.005 * (0.20 + 0.75 * latent_users.satisfaction)
    ).clamp(0.02, 0.95)
    leave_probability = torch.sigmoid(
        -3.2 + 2.4 * latent_users.fatigue
        - 1.7 * latent_users.satisfaction
        + 1.4 * negative.float()
    )
    leave = uniform(users.user_id, step, 317, seed) < leave_probability
    users.active &= ~leave
