"""Adapters from existing reusable ranker blocks to the twin task contract."""

from __future__ import annotations

import torch
from torch import nn

from ....multitask import MultiGateMixtureOfExperts
from ....poi_distribution.models.architectures import CrossLayer


class WideDeepNetwork(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: int):
        super().__init__()
        self.wide = nn.Linear(inputs, outputs)
        self.deep = nn.Sequential(
            nn.Linear(inputs, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, values):
        return self.wide(values) + self.deep(values)


class DCNv2Network(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: int):
        super().__init__()
        self.cross = nn.ModuleList(CrossLayer(inputs) for _ in range(3))
        self.deep = nn.Sequential(
            nn.Linear(inputs, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(inputs + hidden, outputs)

    def forward(self, values):
        crossed = values
        for layer in self.cross:
            crossed = layer(values, crossed)
        return self.head(torch.cat((crossed, self.deep(values)), dim=1))


class MMoENetwork(nn.Module):
    def __init__(
        self, inputs: int, tasks: tuple[str, ...], hidden: int,
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.mmoe = MultiGateMixtureOfExperts(
            inputs, tasks, expert_count=6, expert_dim=hidden
        )

    def forward(self, values):
        output = self.mmoe(values)
        return torch.stack(tuple(output[task] for task in self.tasks), dim=1)


def build_network(
    architecture: str,
    inputs: int,
    tasks: tuple[str, ...],
    hidden: int,
) -> nn.Module:
    outputs = len(tasks)
    if architecture == "lr":
        return nn.Linear(inputs, outputs)
    if architecture == "mlp":
        return nn.Sequential(
            nn.Linear(inputs, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, outputs),
        )
    if architecture == "wide_deep":
        return WideDeepNetwork(inputs, outputs, hidden)
    if architecture == "dcnv2":
        return DCNv2Network(inputs, outputs, hidden)
    if architecture == "mmoe":
        return MMoENetwork(inputs, tasks, hidden)
    raise ValueError(f"unsupported ranker architecture: {architecture}")
