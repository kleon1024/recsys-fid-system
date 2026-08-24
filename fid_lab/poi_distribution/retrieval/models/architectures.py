"""ANN-compatible Two-Tower and multi-interest retrieval models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class TwoTowerRetriever(nn.Module):
    name = "two_tower"

    def __init__(self, query_dim=32, item_dim=23, embedding_dim=32):
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, 64), nn.ReLU(), nn.Linear(64, embedding_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 64), nn.ReLU(), nn.Linear(64, embedding_dim)
        )

    def encode_query(self, values):
        return functional.normalize(self.query_tower(values), dim=-1)

    def encode_item(self, values):
        return functional.normalize(self.item_tower(values), dim=-1)

    def pool_scores(self, query, items):
        return torch.einsum("bd,bkd->bk", self.encode_query(query), items)


class MultiInterestRetriever(TwoTowerRetriever):
    name = "multi_interest"

    def __init__(self, query_dim=32, item_dim=23, embedding_dim=32, interests=3):
        super().__init__(query_dim, item_dim, embedding_dim)
        self.interests = interests
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim, 96), nn.ReLU(),
            nn.Linear(96, embedding_dim * interests),
        )

    def encode_query(self, values):
        states = self.query_tower(values).reshape(len(values), self.interests, -1)
        return functional.normalize(states, dim=-1)

    def pool_scores(self, query, items):
        scores = torch.einsum("bid,bkd->bik", self.encode_query(query), items)
        return scores.max(dim=1).values


def build_retriever(name, query_dim=32, item_dim=23):
    if name == "two_tower":
        return TwoTowerRetriever(query_dim, item_dim)
    if name == "multi_interest":
        return MultiInterestRetriever(query_dim, item_dim)
    raise ValueError(f"unknown POI retriever: {name}")
