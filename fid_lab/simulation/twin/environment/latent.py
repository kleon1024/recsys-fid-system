"""Private user and item truth that cannot cross into platform features."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


def _clone(value):
    return type(value)(**{
        field.name: getattr(value, field.name).clone()
        for field in fields(value)
    })


@dataclass
class LatentUserState:
    long_interest: torch.Tensor
    short_interest: torch.Tensor
    satisfaction: torch.Tensor
    fatigue: torch.Tensor
    conformity: torch.Tensor
    spending_power: torch.Tensor
    commerce_intent: torch.Tensor
    local_intent: torch.Tensor
    creator_intent: torch.Tensor
    activity_propensity: torch.Tensor
    surface_intent: torch.Tensor
    signup_step: torch.Tensor
    retained: torch.Tensor
    habit_strength: torch.Tensor

    def clone(self):
        return _clone(self)


@dataclass
class LatentCatalogState:
    semantic_embedding: torch.Tensor
    true_quality: torch.Tensor
    true_risk: torch.Tensor
    price_appeal: torch.Tensor

    def clone(self):
        return _clone(self)


def select_latent(value, selector):
    return type(value)(**{
        field.name: getattr(value, field.name)[selector].clone()
        for field in fields(value)
    })


def writeback_latent(target, source, selector) -> None:
    for field in fields(target):
        getattr(target, field.name)[selector] = getattr(source, field.name)
