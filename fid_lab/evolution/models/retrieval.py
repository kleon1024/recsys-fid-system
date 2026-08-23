"""Two-tower and multi-interest retrieval representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class RetrievalSnapshot:
    """CPU serving artifact exported from the trained query and item towers."""

    version: str
    query_hidden_weight: np.ndarray
    query_hidden_bias: np.ndarray
    query_output_weight: np.ndarray
    query_output_bias: np.ndarray
    item_embeddings: np.ndarray

    def encode_query(self, query: np.ndarray) -> np.ndarray:
        hidden = np.maximum(
            np.asarray(query) @ self.query_hidden_weight.T + self.query_hidden_bias,
            0.0,
        )
        state = hidden @ self.query_output_weight.T + self.query_output_bias
        return state / max(float(np.linalg.norm(state)), 1e-8)

    def scores(self, query: np.ndarray) -> np.ndarray:
        return self.item_embeddings @ self.encode_query(query)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            version=np.asarray(self.version),
            query_hidden_weight=self.query_hidden_weight,
            query_hidden_bias=self.query_hidden_bias,
            query_output_weight=self.query_output_weight,
            query_output_bias=self.query_output_bias,
            item_embeddings=self.item_embeddings,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalSnapshot":
        with np.load(path) as values:
            return cls(
                str(values["version"]),
                values["query_hidden_weight"],
                values["query_hidden_bias"],
                values["query_output_weight"],
                values["query_output_bias"],
                values["item_embeddings"],
            )


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

    def export_snapshot(
        self,
        items: torch.Tensor,
        version: str,
    ) -> RetrievalSnapshot:
        self.eval()
        with torch.no_grad():
            item_embeddings = self.encode_item(items.to(next(self.parameters()).device))
        first = self.query_tower[0]
        output = self.query_tower[2]
        return RetrievalSnapshot(
            version,
            first.weight.detach().cpu().numpy().copy(),
            first.bias.detach().cpu().numpy().copy(),
            output.weight.detach().cpu().numpy().copy(),
            output.bias.detach().cpu().numpy().copy(),
            item_embeddings.cpu().numpy().copy(),
        )


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
