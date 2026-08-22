"""Comparable CTR baselines sharing exactly one encoded feature batch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .schema import FeatureRegistry


class SparseInput(nn.Module):
    def __init__(self, bucket_sizes: Sequence[int], embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in bucket_sizes
        )
        self.linear = nn.ModuleList(nn.Embedding(size, 1) for size in bucket_sizes)
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        for linear in self.linear:
            nn.init.zeros_(linear.weight)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)], dim=1
        )

    def wide(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [linear(x[:, i]).squeeze(-1) for i, linear in enumerate(self.linear)], dim=1
        ).sum(dim=1)


def mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(0.05)])
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class _WideInteractionModel(nn.Module):
    def __init__(self, bucket_sizes: Sequence[int], embedding_dim: int = 8) -> None:
        super().__init__()
        self.sparse = SparseInput(bucket_sizes, embedding_dim)
        self.deep = mlp(len(bucket_sizes) * embedding_dim, (64, 32), 1)


class WideDeep(_WideInteractionModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.sparse.embed(x)
        return self.sparse.wide(x) + self.deep(embeddings.flatten(1)).squeeze(-1)


class DeepFM(_WideInteractionModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.sparse.embed(x)
        summed = embeddings.sum(dim=1)
        # FM identity computes all pairwise dot products in O(fields * dim).
        fm = 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(dim=1)
        deep = self.deep(embeddings.flatten(1)).squeeze(-1)
        return self.sparse.wide(x) + fm + deep


class ThreeTower(nn.Module):
    """A declared user/item/context ranking model, not a universal named architecture."""

    def __init__(
        self, bucket_sizes: Sequence[int], registry: FeatureRegistry, embedding_dim: int = 8
    ) -> None:
        super().__init__()
        self.sparse = SparseInput(bucket_sizes, embedding_dim)
        self.user_indices = registry.indices_by_group("user")
        self.item_indices = registry.indices_by_group("item")
        self.context_indices = registry.indices_by_group("context") + registry.indices_by_group("cross")
        tower_dim = 16
        self.user_tower = mlp(len(self.user_indices) * embedding_dim, (32,), tower_dim)
        self.item_tower = mlp(len(self.item_indices) * embedding_dim, (32,), tower_dim)
        self.context_tower = mlp(len(self.context_indices) * embedding_dim, (32,), tower_dim)
        self.head = mlp(tower_dim * 3 + 1, (32,), 1)

    @staticmethod
    def select(embeddings: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return embeddings[:, indices, :].flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.sparse.embed(x)
        user = self.user_tower(self.select(embeddings, self.user_indices))
        item = self.item_tower(self.select(embeddings, self.item_indices))
        context = self.context_tower(self.select(embeddings, self.context_indices))
        retrieval_score = (user * item).sum(dim=1, keepdim=True) / user.shape[1] ** 0.5
        return self.head(torch.cat([user, item, context, retrieval_score], dim=1)).squeeze(-1)
