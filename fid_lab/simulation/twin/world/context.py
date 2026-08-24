"""Global, country, regional, temporal, and hotspot context state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...randomness.counter import uniform
from ..contracts import TwinConfig


@dataclass
class ContextState:
    global_topic_heat: torch.Tensor
    country_topic_heat: torch.Tensor
    region_activity: torch.Tensor
    global_hotspot_topic: torch.Tensor
    country_hotspot_topic: torch.Tensor

    def clone(self):
        return ContextState(
            self.global_topic_heat.clone(),
            self.country_topic_heat.clone(),
            self.region_activity.clone(),
            self.global_hotspot_topic.clone(),
            self.country_hotspot_topic.clone(),
        )


def initialize_context(config: TwinConfig, device):
    topic = torch.arange(config.topics, device=device)
    country = torch.arange(config.countries, device=device)
    region = torch.arange(
        config.countries * config.regions_per_country, device=device
    )
    global_heat = 0.20 + 0.80 * uniform(topic, 0, 401, config.seed)
    country_heat = 0.15 + 0.85 * uniform(
        country, 0, 409, config.seed, config.topics
    )
    return ContextState(
        global_topic_heat=global_heat / global_heat.sum(),
        country_topic_heat=country_heat / country_heat.sum(dim=1, keepdim=True),
        region_activity=0.25 + 0.75 * uniform(
            region, 0, 419, config.seed
        ),
        global_hotspot_topic=torch.remainder(
            torch.tensor(config.seed, device=device), config.topics
        ),
        country_hotspot_topic=torch.remainder(
            country * 7_919 + config.seed, config.topics
        ),
    )


def advance_context(context: ContextState, config: TwinConfig, step: int):
    device = context.global_topic_heat.device
    topic = torch.remainder(
        torch.tensor(step * 7_919 + config.seed, device=device), config.topics
    )
    context.global_hotspot_topic = topic
    shock = torch.zeros_like(context.global_topic_heat)
    shock[topic] = 1.0
    context.global_topic_heat = 0.86 * context.global_topic_heat + 0.14 * shock
    context.global_topic_heat /= context.global_topic_heat.sum()
    country = torch.arange(config.countries, device=device)
    country_topic = torch.remainder(
        country * 48_271 + step * 503 + config.seed, config.topics
    )
    context.country_hotspot_topic = country_topic
    country_shock = torch.zeros_like(context.country_topic_heat)
    country_shock[country, country_topic] = 1.0
    context.country_topic_heat = (
        0.90 * context.country_topic_heat + 0.10 * country_shock
    )
    context.country_topic_heat /= context.country_topic_heat.sum(
        dim=1, keepdim=True
    )
