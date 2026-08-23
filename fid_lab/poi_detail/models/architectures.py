"""Separate related-POI, product, and review ranker families."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn

from ...multitask import MultiGateMixtureOfExperts
from ..contracts import DETAIL_TASKS


class LinearModule(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        del semantic_dim
        self.heads = nn.ModuleDict({
            task: nn.Linear(width, 1) for task in DETAIL_TASKS
        })

    def forward(self, features, semantic, history):
        del semantic, history
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        return {task: head(flat).reshape(shape) for task, head in self.heads.items()}


class WideDeepModule(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        del semantic_dim
        self.wide = nn.ModuleDict({
            task: nn.Linear(width, 1) for task in DETAIL_TASKS
        })
        self.deep = nn.Sequential(
            nn.Linear(width, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(24, 1) for task in DETAIL_TASKS
        })

    def forward(self, features, semantic, history):
        del semantic, history
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        deep = self.deep(flat)
        return {
            task: (self.wide[task](flat) + self.heads[task](deep)).reshape(shape)
            for task in DETAIL_TASKS
        }


class RelatedDINModule(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        self.query = nn.Linear(semantic_dim, semantic_dim)
        self.key = nn.Linear(semantic_dim, semantic_dim)
        self.shared = nn.Sequential(
            nn.Linear(width + 2 * semantic_dim, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(24, 1) for task in DETAIL_TASKS
        })
        self.scale = sqrt(semantic_dim)

    def forward(self, features, semantic, history):
        attention = torch.softmax(
            torch.einsum(
                "bkd,bld->bkl", self.query(semantic), self.key(history)
            ) / self.scale,
            dim=2,
        )
        pooled = torch.einsum("bkl,bld->bkd", attention, history)
        shape = features.shape[:2]
        state = self.shared(torch.cat(
            (features, semantic, pooled), dim=2
        ).flatten(0, 1))
        return {task: head(state).reshape(shape) for task, head in self.heads.items()}


class ProductMMoEModule(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        del semantic_dim
        self.network = MultiGateMixtureOfExperts(width, DETAIL_TASKS, 4, 32)

    def forward(self, features, semantic, history):
        del semantic, history
        shape = features.shape[:2]
        outputs = self.network(features.flatten(0, 1))
        return {
            name: value.reshape(*shape, -1) if name.startswith("gate:")
            else value.reshape(shape)
            for name, value in outputs.items()
        }


class ReviewDeepModule(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        del semantic_dim
        self.encoder = nn.Sequential(
            nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 24), nn.GELU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(24, 1) for task in DETAIL_TASKS
        })

    def forward(self, features, semantic, history):
        del semantic, history
        shape = features.shape[:2]
        state = self.encoder(features.flatten(0, 1))
        return {task: head(state).reshape(shape) for task, head in self.heads.items()}


class ModuleFamily(nn.Module):
    def __init__(self, width, semantic_dim, builders):
        super().__init__()
        self.modules_by_kind = nn.ModuleList([
            builder(width, semantic_dim) for builder in builders
        ])

    def forward(self, features, semantic, history, module_kind):
        module_outputs = [
            module(features, semantic, history) for module in self.modules_by_kind
        ]
        return {
            task: sum(
                output[task] * (module_kind == kind).float()
                for kind, output in enumerate(module_outputs)
            )
            for task in DETAIL_TASKS
        }


def build_family(name, width, semantic_dim):
    builders = {
        "linear": (LinearModule,) * 3,
        "wide_deep": (WideDeepModule,) * 3,
        "specialized": (RelatedDINModule, ProductMMoEModule, ReviewDeepModule),
    }
    return ModuleFamily(width, semantic_dim, builders[name])


FAMILY_NAMES = ("linear", "wide_deep", "specialized")
