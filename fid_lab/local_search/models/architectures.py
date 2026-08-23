"""Linear, Wide & Deep, DIN, and Transformer+MMoE search rankers."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn

from ...multitask import MultiGateMixtureOfExperts
from ...training.common.request_rankers import RequestLinearRanker
from ..contracts import LOCAL_SEARCH_TASKS


class LinearRanker(RequestLinearRanker):
    def __init__(self, width, semantic_dim):
        del semantic_dim
        super().__init__(width, LOCAL_SEARCH_TASKS)


class WideDeepRanker(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        del semantic_dim
        self.wide = nn.ModuleDict({
            task: nn.Linear(width, 1) for task in LOCAL_SEARCH_TASKS
        })
        self.deep = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(32, 1) for task in LOCAL_SEARCH_TASKS
        })

    def forward(self, features, candidate_semantic, history):
        del candidate_semantic, history
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        deep = self.deep(flat)
        return {
            task: (self.wide[task](flat) + self.heads[task](deep)).reshape(shape)
            for task in LOCAL_SEARCH_TASKS
        }


class DINRanker(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        self.query = nn.Linear(semantic_dim, semantic_dim)
        self.key = nn.Linear(semantic_dim, semantic_dim)
        self.shared = nn.Sequential(
            nn.Linear(width + 2 * semantic_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(32, 1) for task in LOCAL_SEARCH_TASKS
        })
        self.scale = sqrt(semantic_dim)

    def forward(self, features, candidate_semantic, history):
        attention = torch.softmax(
            torch.einsum(
                "bkd,bld->bkl", self.query(candidate_semantic), self.key(history)
            ) / self.scale,
            dim=2,
        )
        pooled = torch.einsum("bkl,bld->bkd", attention, history)
        shape = features.shape[:2]
        state = self.shared(torch.cat(
            (features, candidate_semantic, pooled), dim=2
        ).flatten(0, 1))
        return {task: head(state).reshape(shape) for task, head in self.heads.items()}


class TransformerMMoERanker(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            semantic_dim, 4, semantic_dim * 2, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.query = nn.Linear(semantic_dim, semantic_dim)
        self.mmoe = MultiGateMixtureOfExperts(
            width + 2 * semantic_dim, LOCAL_SEARCH_TASKS, 4, 32
        )
        self.scale = sqrt(semantic_dim)

    def forward(self, features, candidate_semantic, history):
        encoded = self.encoder(history)
        attention = torch.softmax(
            torch.einsum(
                "bkd,bld->bkl", self.query(candidate_semantic), encoded
            ) / self.scale,
            dim=2,
        )
        pooled = torch.einsum("bkl,bld->bkd", attention, encoded)
        outputs = self.mmoe(torch.cat(
            (features, candidate_semantic, pooled), dim=2
        ).flatten(0, 1))
        shape = features.shape[:2]
        return {
            name: value.reshape(*shape, -1) if name.startswith("gate:")
            else value.reshape(shape)
            for name, value in outputs.items()
        }


MODEL_FACTORIES = {
    "linear": LinearRanker,
    "wide_deep": WideDeepRanker,
    "din": DINRanker,
    "transformer_mmoe": TransformerMMoERanker,
}
