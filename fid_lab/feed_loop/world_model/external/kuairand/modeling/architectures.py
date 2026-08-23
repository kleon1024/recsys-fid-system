"""Multi-behavior W&D and sequence Transformer over KuaiRand contracts."""

from __future__ import annotations

import torch
from torch import nn


TASK_COUNT = 8


class SparseContext(nn.Module):
    def __init__(self, vocabularies, embedding_dim=16, width=64) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(vocabulary, embedding_dim, padding_idx=0)
            for vocabulary in vocabularies
        )
        self.projection = nn.Sequential(
            nn.Linear(len(vocabularies) * embedding_dim, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )

    def forward(self, sparse):
        values = [
            embedding(sparse[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.projection(torch.cat(values, dim=1))


class KuaiWideDeep(nn.Module):
    def __init__(self, vocabularies, dense_dim, width=64) -> None:
        super().__init__()
        self.sparse = SparseContext(vocabularies, width=width)
        self.wide = nn.Linear(dense_dim, TASK_COUNT)
        self.deep = nn.Sequential(
            nn.Linear(width + dense_dim, 128), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(128, width), nn.SiLU(), nn.Linear(width, TASK_COUNT),
        )

    def forward(self, sparse, dense, history_items=None, history_feedback=None):
        del history_items, history_feedback
        context = self.sparse(sparse)
        return self.wide(dense) + self.deep(torch.cat((context, dense), dim=1))


class KuaiSequenceTransformer(nn.Module):
    def __init__(self, vocabularies, dense_dim, sequence_length, width=64) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.sparse = SparseContext(vocabularies, width=width)
        self.item_projection = nn.Linear(16, width)
        self.feedback_projection = nn.Linear(7, width)
        self.position = nn.Embedding(sequence_length, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=0.10, batch_first=True, norm_first=True
        )
        self.history_encoder = nn.TransformerEncoder(layer, 2)
        self.interest_attention = nn.MultiheadAttention(
            width, 4, dropout=0.05, batch_first=True
        )
        self.current = nn.Sequential(
            nn.Linear(width + dense_dim, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.head = nn.Sequential(
            nn.Linear(width * 3, 128), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(128, TASK_COUNT),
        )

    def encode_current(self, sparse, dense):
        return self.current(torch.cat((self.sparse(sparse), dense), dim=1))

    def encode_history(self, history_items, history_feedback):
        item_embedding = self.sparse.embeddings[1](history_items)
        positions = self.position(
            torch.arange(self.sequence_length, device=history_items.device)
        )[None]
        history = self.item_projection(item_embedding)
        history += self.feedback_projection(history_feedback.float()) + positions
        padding = history_items == 0
        all_empty = padding.all(dim=1)
        padding = padding.clone()
        padding[all_empty, 0] = False
        encoded = self.history_encoder(history, src_key_padding_mask=padding)
        return encoded, padding

    def score_encoded(self, current, encoded, padding):
        interest, _ = self.interest_attention(
            current, encoded, encoded, key_padding_mask=padding,
            need_weights=False,
        )
        return self.head(torch.cat((current, interest, current * interest), dim=-1))

    def score_slate(self, sparse, dense, history_items, history_feedback):
        batch, candidates, fields = sparse.shape
        current = self.encode_current(
            sparse.reshape(-1, fields), dense.reshape(-1, dense.shape[-1])
        ).reshape(batch, candidates, -1)
        encoded, padding = self.encode_history(history_items, history_feedback)
        return self.score_encoded(current, encoded, padding)

    def forward(self, sparse, dense, history_items, history_feedback):
        current = self.encode_current(sparse, dense)
        encoded, padding = self.encode_history(history_items, history_feedback)
        return self.score_encoded(current[:, None], encoded, padding)[:, 0]


class KuaiSequenceMMoE(KuaiSequenceTransformer):
    def __init__(self, vocabularies, dense_dim, sequence_length, width=64,
                 experts=4) -> None:
        super().__init__(vocabularies, dense_dim, sequence_length, width)
        representation = width * 3
        self.head = nn.Identity()
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(representation, 128), nn.SiLU(),
                nn.Linear(128, width), nn.SiLU(),
            )
            for _ in range(experts)
        )
        self.gates = nn.ModuleList(
            nn.Linear(representation, experts) for _ in range(TASK_COUNT)
        )
        self.towers = nn.ModuleList(
            nn.Linear(width, 1) for _ in range(TASK_COUNT)
        )

    def score_encoded(self, current, encoded, padding):
        interest, _ = self.interest_attention(
            current, encoded, encoded, key_padding_mask=padding,
            need_weights=False,
        )
        representation = torch.cat(
            (current, interest, current * interest), dim=-1
        )
        expert = torch.stack(
            tuple(module(representation) for module in self.experts), dim=-2
        )
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weight = torch.softmax(gate(representation), dim=-1)
            mixture = (expert * weight[..., None]).sum(dim=-2)
            outputs.append(tower(mixture))
        return torch.cat(outputs, dim=-1)
