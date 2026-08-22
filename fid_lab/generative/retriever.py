"""Constrained beam decoding over a versioned Semantic-ID prefix trie."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantic_ids import SemanticIdIndex


@dataclass(frozen=True)
class GeneratedItem:
    item_id: int
    semantic_id: tuple[int, ...]
    score: float


class GenerativeRetriever:
    def __init__(self, index: SemanticIdIndex) -> None:
        self.index = index
        self.embedding_by_item = {
            int(item_id): index.embeddings[row]
            for row, item_id in enumerate(index.item_ids)
        }

    def prefix_score(self, query: np.ndarray, prefix: tuple[int, ...]) -> float:
        matching = [
            item_id for item_id, code in self.index.codes.items() if code[: len(prefix)] == prefix
        ]
        if not matching:
            return float("-inf")
        # Local teaching scorer: best compatible item similarity under the valid prefix.
        return max(float(query @ self.embedding_by_item[item_id]) for item_id in matching)

    def retrieve(self, query: np.ndarray, limit: int = 20, beam_size: int = 40) -> list[GeneratedItem]:
        code_length = len(next(iter(self.index.codes.values())))
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for _ in range(code_length):
            expanded: list[tuple[tuple[int, ...], float]] = []
            for prefix, _ in beams:
                for token in self.index.valid_next_tokens(prefix):
                    next_prefix = prefix + (token,)
                    expanded.append((next_prefix, self.prefix_score(query, next_prefix)))
            beams = sorted(expanded, key=lambda value: (-value[1], value[0]))[:beam_size]
        generated: list[GeneratedItem] = []
        for code, score in beams:
            item_id = self.index.item_for_code(code)
            if item_id is not None:
                generated.append(GeneratedItem(item_id, code, score))
        return generated[:limit]
