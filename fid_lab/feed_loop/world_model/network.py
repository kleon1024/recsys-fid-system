"""Partially observed variational slate-and-sequence world model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import STOCHASTIC_ACTIONS, WorldModelConfig


@dataclass
class WorldModelOutput:
    logits: dict[str, torch.Tensor]
    stay_mixture_logits: torch.Tensor
    stay_mean: torch.Tensor
    stay_log_scale: torch.Tensor
    context: torch.Tensor


class NeuralSCM(nn.Module):
    """Neural SCM whose latent state is never exposed as a serving feature."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("stay_calibration_shift", torch.zeros(()))
        self.register_buffer("stay_calibration_scale", torch.ones(()))
        width = config.width
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, width),
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
        )
        self.sequence_encoder = nn.GRU(
            config.sequence_dim, width, batch_first=True
        )
        self.slate_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.lifecycle_embedding = nn.Embedding(4, 8)
        self.region_embedding = nn.Embedding(10, 8)
        context_inputs = width * 3 + 16
        self.context_projection = nn.Sequential(
            nn.Linear(context_inputs, width * 2), nn.SiLU(), nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width), nn.SiLU(),
        )
        self.prior = nn.Linear(width, config.latent_dim * 2)
        decoder_width = width + config.latent_dim
        self.decoder_init = nn.Sequential(
            nn.Linear(decoder_width, width), nn.SiLU()
        )
        self.stay_head = nn.Linear(width, config.stay_mixture_components * 3)
        self.action_heads = nn.ModuleDict(
            {action.name: nn.Linear(width, 1) for action in STOCHASTIC_ACTIONS}
        )
        self.action_transition = nn.GRUCell(2, width)
        self.event_transition = nn.GRUCell(config.sequence_dim, width)

    def encode_context(self, selected, slate, sequence, lifecycle, region):
        selected_hidden = self.feature_encoder(selected)
        slate_hidden = self.feature_encoder(slate)
        attended, _ = self.slate_attention(
            selected_hidden[:, None], slate_hidden, slate_hidden, need_weights=False
        )
        _, sequence_hidden = self.sequence_encoder(sequence)
        cohort = torch.cat((
            self.lifecycle_embedding(lifecycle.clamp(0, 3)),
            self.region_embedding(region.clamp(0, 9)),
        ), dim=1)
        return self.context_projection(torch.cat((
            selected_hidden, attended[:, 0], sequence_hidden[-1], cohort,
        ), dim=1))

    @staticmethod
    def _latent(parameters, noise):
        mean, log_var = parameters.chunk(2, dim=1)
        log_var = log_var.clamp(-6.0, 3.0)
        return mean + torch.exp(0.5 * log_var) * noise, mean, log_var

    def forward(self, batch, latent_noise=None) -> WorldModelOutput:
        context = self.encode_context(
            batch["selected_features"], batch["slate_features"], batch["sequence"],
            batch["lifecycle"], batch["region"],
        )
        prior_parameters = self.prior(context)
        if latent_noise is None:
            latent_noise = torch.randn(
                len(context), self.config.latent_dim, device=context.device
            )
        latent, _, _ = self._latent(prior_parameters, latent_noise)
        hidden = self.decoder_init(torch.cat((context, latent), dim=1))
        stay_parameters = self.stay_head(hidden).reshape(
            len(hidden), self.config.stay_mixture_components, 3
        )
        stay_mixture_logits = stay_parameters[:, :, 0]
        stay_mean = (
            torch.sigmoid(stay_parameters[:, :, 1]) * self.stay_calibration_scale
            + self.stay_calibration_shift
        ).clamp(0.0, 1.0)
        stay_log_scale = (
            stay_parameters[:, :, 2].clamp(-4.0, 0.5)
            + torch.log(self.stay_calibration_scale.clamp_min(1e-4))
        )
        logits = {}
        for action in STOCHASTIC_ACTIONS:
            logit = self.action_heads[action.name](hidden).squeeze(1)
            logits[action.name] = logit
            action_value = torch.sigmoid(logit)
            hidden = self.action_transition(
                torch.stack((action_value, torch.sigmoid(logit)), dim=1), hidden
            )
        return WorldModelOutput(
            logits, stay_mixture_logits, stay_mean, stay_log_scale, context,
        )

    def next_hidden(self, context: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        return self.event_transition(event, context)

    def sample(self, batch, latent_noise, mixture_uniform, stay_noise, action_uniforms):
        context = self.encode_context(
            batch["selected_features"], batch["slate_features"], batch["sequence"],
            batch["lifecycle"], batch["region"],
        )
        latent, _, _ = self._latent(self.prior(context), latent_noise)
        hidden = self.decoder_init(torch.cat((context, latent), dim=1))
        stay_parameters = self.stay_head(hidden).reshape(
            len(hidden), self.config.stay_mixture_components, 3
        )
        mixture_probability = torch.softmax(stay_parameters[:, :, 0], dim=1)
        mixture_choice = (
            mixture_uniform[:, None] > mixture_probability.cumsum(dim=1)
        ).sum(dim=1).clamp_max(self.config.stay_mixture_components - 1)
        rows = torch.arange(len(hidden), device=hidden.device)
        stay_mean = (
            torch.sigmoid(stay_parameters[rows, mixture_choice, 1])
            * self.stay_calibration_scale
            + self.stay_calibration_shift
        ).clamp(0.0, 1.0)
        stay_scale = torch.exp(
            stay_parameters[rows, mixture_choice, 2].clamp(-4.0, 0.5)
        ) * self.stay_calibration_scale
        normalized_stay = (stay_mean + stay_scale * stay_noise).clamp(0.0, 1.0)
        duration = torch.expm1(
            batch["selected_features"][:, 12].clamp(0.0, 1.0) * torch.log(
                torch.tensor(181.0, device=context.device)
            )
        ).clamp(1.0, 180.0)
        stay = torch.minimum(torch.expm1(normalized_stay * torch.log(
            torch.tensor(181.0, device=context.device)
        )), duration)
        actions = {}
        probabilities = {}
        for index, action in enumerate(STOCHASTIC_ACTIONS):
            logit = self.action_heads[action.name](hidden).squeeze(1)
            probability = torch.sigmoid(logit)
            sampled = action_uniforms[:, index] < probability
            if action.requires is not None:
                sampled &= actions[action.requires]
            actions[action.name] = sampled
            probabilities[action.name] = probability
            hidden = self.action_transition(
                torch.stack((sampled.float(), probability), dim=1), hidden
            )
        stay *= actions["play"].float()
        completion = (stay / duration).clamp(0.0, 1.0)
        actions["play_3s"] = actions["play"] & (stay >= 3.0)
        actions["complete_play"] = actions["play"] & (completion >= 0.95)
        actions["long_view"] = actions["play"] & (
            stay >= torch.minimum(torch.full_like(stay, 18.0), duration)
        )
        actions["quality_long_view"] = actions["play"] & (
            stay >= torch.minimum(torch.full_like(stay, 30.0), duration)
        )
        event = torch.stack((
            batch["selected_features"][:, 17],
            torch.log1p(stay) / torch.log(torch.tensor(181.0, device=context.device)),
            actions["long_view"].float(), actions["quality_long_view"].float(),
            actions["like"].float(), actions["negative_feedback"].float(),
            actions["anchor_click"].float(), actions["conversion"].float(),
        ), dim=1)
        return {
            "actions": actions,
            "probabilities": probabilities,
            "stay_seconds": stay,
            "completion": completion,
            "event": event,
            "next_hidden": self.next_hidden(context, event),
        }
