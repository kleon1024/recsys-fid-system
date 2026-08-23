"""Propensity-corrected variational multi-action world-model objective."""

from __future__ import annotations

import math

import torch
from torch import nn

from .contracts import STOCHASTIC_ACTIONS, STAY_LABEL_INDEX, WorldModelConfig


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _stay_nll(output, labels):
    stayed = labels[:, STAY_LABEL_INDEX].clamp_min(0.0)
    played = labels[:, 0]
    target = torch.log1p(stayed) / math.log(181.0)
    scale = torch.exp(output.stay_log_scale).clamp_min(1e-3)
    normal = torch.distributions.Normal(output.stay_mean, scale)
    mixture_log_probability = torch.log_softmax(
        output.stay_mixture_logits, dim=1
    )
    duration = labels[:, STAY_LABEL_INDEX] / labels[:, 3].clamp_min(1e-4)
    duration_target = torch.log1p(duration.clamp(0.0, 180.0)) / math.log(181.0)
    censored = (labels[:, 3] >= 0.995) & (played > 0.5)
    density = -torch.logsumexp(
        mixture_log_probability + normal.log_prob(target[:, None]), dim=1
    )
    survival = -torch.logsumexp(
        mixture_log_probability
        + torch.log((1.0 - normal.cdf(duration_target[:, None])).clamp_min(1e-7)),
        dim=1,
    )
    return torch.where(censored, survival, density) * played


def world_model_loss(output, batch, config: WorldModelConfig):
    weights = batch["weights"].clamp_max(config.max_ips_weight)
    task_losses = {}
    total = torch.zeros(len(weights), device=weights.device)
    for action in STOCHASTIC_ACTIONS:
        target = batch["labels"][:, action.label_index]
        loss = nn.functional.binary_cross_entropy_with_logits(
            output.logits[action.name], target, reduction="none"
        )
        task_weights = weights * batch["label_masks"][:, action.label_index]
        task_losses[action.name] = float(_weighted_mean(loss.detach(), task_weights))
        total += loss * batch["label_masks"][:, action.label_index]
    stay = _stay_nll(output, batch["labels"])
    total += config.stay_loss_weight * stay
    return _weighted_mean(total, weights), {
        **task_losses,
        "stay_nll": float(_weighted_mean(stay.detach(), weights)),
    }
