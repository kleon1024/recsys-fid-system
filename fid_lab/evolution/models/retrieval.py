"""Two-tower and multi-interest retrieval representations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class TwoTowerRetriever(nn.Module):
    def __init__(self, query_dim: int, item_dim: int, representation_dim: int = 32) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, 64), nn.ReLU(), nn.Linear(64, representation_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 64), nn.ReLU(), nn.Linear(64, representation_dim)
        )

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.query_tower(query), dim=1)

    def encode_item(self, item: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.item_tower(item), dim=1)

    def forward(self, query: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return self.encode_query(query) @ self.encode_item(item).T


class MultiInterestTwoTower(TwoTowerRetriever):
    def __init__(
        self,
        query_dim: int,
        item_dim: int,
        representation_dim: int = 32,
        interests: int = 3,
    ) -> None:
        super().__init__(query_dim, item_dim, representation_dim)
        self.interests = interests
        self.multi_query = nn.Linear(query_dim, representation_dim * interests)

    def encode_interests(self, query: torch.Tensor) -> torch.Tensor:
        states = self.multi_query(query).reshape(len(query), self.interests, -1)
        return functional.normalize(states, dim=2)

    def forward(self, query: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        interests = self.encode_interests(query)
        items = self.encode_item(item)
        scores = torch.einsum("bkd,nd->bkn", interests, items)
        return scores.max(dim=1).values
