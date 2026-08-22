"""Learned Semantic-ID autoregressive retrieval and constrained session selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ...generative.semantic_ids import SemanticIdIndex


@dataclass(frozen=True)
class SemanticVocabulary:
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    total_tokens: int

    @classmethod
    def from_index(cls, index: SemanticIdIndex) -> "SemanticVocabulary":
        codes = tuple(index.codes.values())
        sizes = tuple(max(code[level] for code in codes) + 1 for level in range(len(codes[0])))
        offsets: list[int] = []
        cursor = 1
        for size in sizes:
            offsets.append(cursor)
            cursor += size
        return cls(tuple(offsets), sizes, cursor)

    def encode(self, code: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.offsets[level] + token for level, token in enumerate(code))


class AutoregressiveSemanticDecoder(nn.Module):
    def __init__(
        self,
        query_dim: int,
        vocabulary: SemanticVocabulary,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.query = nn.Linear(query_dim, hidden)
        self.tokens = nn.Embedding(vocabulary.total_tokens, hidden)
        self.positions = nn.Embedding(len(vocabulary.sizes), hidden)
        layer = nn.TransformerDecoderLayer(hidden, 4, hidden * 2, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.output = nn.Linear(hidden, vocabulary.total_tokens)

    def forward(self, query: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_tokens.shape[1], device=input_tokens.device)
        target = self.tokens(input_tokens) + self.positions(positions)[None, :, :]
        length = input_tokens.shape[1]
        mask = torch.triu(torch.ones(length, length, device=query.device, dtype=torch.bool), 1)
        memory = self.query(query)[:, None, :]
        return self.output(self.decoder(target, memory, tgt_mask=mask))

    def loss(self, query: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        start = torch.zeros((len(codes), 1), dtype=torch.long, device=codes.device)
        inputs = torch.cat([start, codes[:, :-1]], dim=1)
        logits = self(query, inputs)
        return nn.functional.cross_entropy(logits.flatten(0, 1), codes.flatten())

    def next_logits(self, query: torch.Tensor, prefix: tuple[int, ...]) -> torch.Tensor:
        encoded = self.vocabulary.encode(prefix) if prefix else ()
        inputs = torch.tensor(((0,) + encoded,), dtype=torch.long, device=query.device)
        return self(query[None, :], inputs)[0, -1]


@dataclass(frozen=True)
class GeneratedCandidate:
    item_id: int
    semantic_id: tuple[int, ...]
    log_probability: float


class LearnedGenerativeRetriever:
    def __init__(
        self,
        index: SemanticIdIndex,
        model: AutoregressiveSemanticDecoder,
    ) -> None:
        self.index = index
        self.model = model

    def _expand(
        self,
        query: torch.Tensor,
        beams: list[tuple[tuple[int, ...], float]],
    ) -> list[tuple[tuple[int, ...], float]]:
        expanded: list[tuple[tuple[int, ...], float]] = []
        for prefix, score in beams:
            log_probabilities = torch.log_softmax(
                self.model.next_logits(query, prefix), dim=0
            )
            level = len(prefix)
            for token in self.index.valid_next_tokens(prefix):
                global_token = self.model.vocabulary.offsets[level] + token
                expanded.append(
                    (prefix + (token,), score + float(log_probabilities[global_token]))
                )
        return expanded

    def retrieve(
        self,
        query: torch.Tensor,
        limit: int = 20,
        beam_size: int = 40,
    ) -> list[GeneratedCandidate]:
        code_length = len(next(iter(self.index.codes.values())))
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        self.model.eval()
        with torch.no_grad():
            for _ in range(code_length):
                expanded = self._expand(query, beams)
                beams = sorted(expanded, key=lambda value: (-value[1], value[0]))[:beam_size]
        candidates = []
        for code, score in beams:
            item_id = self.index.item_for_code(code)
            if item_id is not None:
                candidates.append(GeneratedCandidate(item_id, code, score))
        return candidates[:limit]


class SessionGenerator:
    def __init__(
        self,
        retriever: LearnedGenerativeRetriever,
        author_by_item: dict[int, int],
        category_by_item: dict[int, int],
    ) -> None:
        self.retriever = retriever
        self.author_by_item = author_by_item
        self.category_by_item = category_by_item

    def generate(
        self,
        query: torch.Tensor,
        size: int,
        author_cap: int = 2,
        category_cap: int = 4,
    ) -> list[GeneratedCandidate]:
        candidates = self.retriever.retrieve(query, limit=size * 5, beam_size=size * 10)
        author_count: dict[int, int] = {}
        category_count: dict[int, int] = {}
        selected: list[GeneratedCandidate] = []
        for candidate in candidates:
            author = self.author_by_item[candidate.item_id]
            category = self.category_by_item[candidate.item_id]
            if author_count.get(author, 0) >= author_cap:
                continue
            if category_count.get(category, 0) >= category_cap:
                continue
            selected.append(candidate)
            author_count[author] = author_count.get(author, 0) + 1
            category_count[category] = category_count.get(category, 0) + 1
            if len(selected) == size:
                break
        return selected


def train_semantic_decoder(
    index: SemanticIdIndex,
    queries: np.ndarray,
    item_ids: np.ndarray,
    epochs: int = 20,
    device: str = "cpu",
) -> tuple[AutoregressiveSemanticDecoder, list[float]]:
    vocabulary = SemanticVocabulary.from_index(index)
    model = AutoregressiveSemanticDecoder(queries.shape[1], vocabulary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    query_tensor = torch.from_numpy(queries.astype(np.float32)).to(device)
    code_tensor = torch.tensor(
        [vocabulary.encode(index.codes[int(item_id)]) for item_id in item_ids],
        dtype=torch.long,
        device=device,
    )
    history = []
    model.train()
    for _ in range(epochs):
        loss = model.loss(query_tensor, code_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return model, history
