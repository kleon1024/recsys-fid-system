"""Linear through MMoE model families for Local multi-task ranking."""

from __future__ import annotations

import torch
from torch import nn

from ...multitask import MultiGateMixtureOfExperts
from ..contracts import TASK_LABELS


OUTPUT_TASKS = tuple(TASK_LABELS)


class LinearRanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.heads = nn.ModuleDict({
            task: nn.Linear(width, 1) for task in OUTPUT_TASKS
        })

    def forward(self, features):
        return {
            task: head(features).squeeze(1) for task, head in self.heads.items()
        }


class WideDeepRanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wide = nn.ModuleDict({
            task: nn.Linear(width, 1) for task in OUTPUT_TASKS
        })
        self.deep = nn.Sequential(
            nn.Linear(width, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(48, 1) for task in OUTPUT_TASKS
        })

    def forward(self, features):
        state = self.deep(features)
        return {
            task: (self.wide[task](features) + self.heads[task](state)).squeeze(1)
            for task in OUTPUT_TASKS
        }


class CrossLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, original, state):
        return original * self.weight(state) + self.bias + state


class DCNv2Ranker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cross = nn.ModuleList(CrossLayer(width) for _ in range(3))
        self.deep = nn.Sequential(
            nn.Linear(width, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(width + 48, 1) for task in OUTPUT_TASKS
        })

    def forward(self, features):
        crossed = features
        for layer in self.cross:
            crossed = layer(features, crossed)
        state = torch.cat((crossed, self.deep(features)), dim=1)
        return {
            task: head(state).squeeze(1) for task, head in self.heads.items()
        }


class MMoERanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = MultiGateMixtureOfExperts(width, OUTPUT_TASKS, 6, 48)

    def forward(self, features):
        return self.network(features)


def build_ranker(name, width):
    factories = {
        "linear": LinearRanker,
        "wide_deep": WideDeepRanker,
        "dcnv2": DCNv2Ranker,
        "mmoe": MMoERanker,
    }
    return factories[name](width)
