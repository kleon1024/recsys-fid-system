"""ANN-compatible user, item, and multi-interest retrieval towers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class ItemTower(nn.Module):
    def __init__(self, vocabularies, width=64, embedding_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocabulary, embedding_dim, padding_idx=0)
            for vocabulary in vocabularies[1:]
        )
        self.projection = nn.Sequential(
            nn.Linear(len(self.embeddings) * embedding_dim + 2, width),
            nn.SiLU(), nn.Linear(width, width),
        )

    def forward(self, sparse, dense):
        categorical = torch.cat(tuple(
            embedding(sparse[:, index + 1])
            for index, embedding in enumerate(self.embeddings)
        ), dim=1)
        item_dense = dense[:, (0, 3)]
        return functional.normalize(
            self.projection(torch.cat((categorical, item_dense), dim=1)), dim=1
        )


class KuaiTwoTowerRetriever(nn.Module):
    def __init__(self, vocabularies, width=64):
        super().__init__()
        self.width = width
        self.item = ItemTower(vocabularies, width)
        self.user = nn.Embedding(vocabularies[0], width, padding_idx=0)
        self.history_item = nn.Embedding(vocabularies[1], width, padding_idx=0)
        self.feedback = nn.Linear(7, 1, bias=False)
        self.context = nn.Sequential(
            nn.Linear(width * 2 + 9, width * 2), nn.SiLU(),
            nn.Linear(width * 2, width),
        )

    def encode_item(self, sparse, dense):
        return self.item(sparse, dense)

    def _history(self, history_items, history_feedback):
        state = self.history_item(history_items)
        padding = history_items == 0
        weight = self.feedback(history_feedback.float()).squeeze(2)
        weight = weight.masked_fill(padding, -torch.inf)
        empty = padding.all(dim=1)
        weight[empty] = 0.0
        attention = torch.softmax(weight, dim=1).masked_fill(padding, 0.0)
        return torch.einsum("bl,bld->bd", attention, state)

    def encode_query(self, sparse, dense, history_items, history_feedback):
        user = self.user(sparse[:, 0])
        history = self._history(history_items, history_feedback)
        request_dense = dense[:, (1, 2, 4, 5, 6, 7, 8, 9, 10)]
        return functional.normalize(
            self.context(torch.cat((user, history, request_dense), dim=1)), dim=1
        )


class KuaiMultiInterestRetriever(KuaiTwoTowerRetriever):
    def __init__(self, vocabularies, width=64, interests=3):
        super().__init__(vocabularies, width)
        self.interests = interests
        self.interest_queries = nn.Parameter(torch.randn(interests, width) * 0.02)
        self.multi_context = nn.Sequential(
            nn.Linear(width * 2 + 9, width * 2), nn.SiLU(),
            nn.Linear(width * 2, width),
        )

    def encode_query(self, sparse, dense, history_items, history_feedback):
        state = self.history_item(history_items)
        padding = history_items == 0
        attention = torch.einsum("kd,bld->bkl", self.interest_queries, state)
        attention = attention.masked_fill(padding[:, None], -torch.inf)
        empty = padding.all(dim=1)
        attention[empty] = 0.0
        attention = torch.softmax(attention, dim=2).masked_fill(
            padding[:, None], 0.0
        )
        interests = torch.einsum("bkl,bld->bkd", attention, state)
        user = self.user(sparse[:, 0])[:, None].expand(-1, self.interests, -1)
        request_dense = dense[:, None, (1, 2, 4, 5, 6, 7, 8, 9, 10)].expand(
            -1, self.interests, -1
        )
        return functional.normalize(
            self.multi_context(torch.cat((user, interests, request_dense), dim=2)),
            dim=2,
        )
