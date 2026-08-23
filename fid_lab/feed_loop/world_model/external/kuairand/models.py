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

    def forward(self, sparse, dense, history_items, history_feedback):
        current = self.current(torch.cat((self.sparse(sparse), dense), dim=1))
        item_embedding = self.sparse.embeddings[1](history_items)
        positions = self.position(
            torch.arange(self.sequence_length, device=sparse.device)
        )[None]
        history = self.item_projection(item_embedding)
        history += self.feedback_projection(history_feedback.float()) + positions
        padding = history_items == 0
        all_empty = padding.all(dim=1)
        padding = padding.clone()
        padding[all_empty, 0] = False
        encoded = self.history_encoder(history, src_key_padding_mask=padding)
        interest, _ = self.interest_attention(
            current[:, None], encoded, encoded, key_padding_mask=padding,
            need_weights=False,
        )
        interest = interest[:, 0]
        return self.head(torch.cat((current, interest, current * interest), dim=1))
