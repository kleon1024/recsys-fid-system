"""Deterministic residual-quantized Semantic IDs with collision suffixes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True, eq=False)
class SemanticIdIndex:
    item_ids: np.ndarray
    embeddings: np.ndarray
    codes: dict[int, tuple[int, ...]]
    codebooks: tuple[np.ndarray, ...]
    version: str = "semantic-id-rq-v1"

    @classmethod
    def fit(
        cls,
        item_ids: np.ndarray,
        embeddings: np.ndarray,
        levels: int = 3,
        codebook_size: int = 8,
        seed: int = 31,
    ) -> "SemanticIdIndex":
        if len(item_ids) != len(embeddings) or len(set(item_ids.tolist())) != len(item_ids):
            raise ValueError("item IDs and embeddings must be aligned and unique")
        residual = embeddings.astype(np.float64).copy()
        level_codes: list[np.ndarray] = []
        codebooks: list[np.ndarray] = []
        for level in range(levels):
            clusters = min(codebook_size, len(item_ids))
            model = KMeans(n_clusters=clusters, random_state=seed + level, n_init=10)
            codes = model.fit_predict(residual)
            residual -= model.cluster_centers_[codes]
            level_codes.append(codes)
            codebooks.append(model.cluster_centers_.copy())
        prefixes = list(zip(*level_codes))
        by_prefix: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for row, prefix in enumerate(prefixes):
            by_prefix[tuple(int(token) for token in prefix)].append(row)
        codes_by_item: dict[int, tuple[int, ...]] = {}
        for prefix, rows in by_prefix.items():
            for collision_index, row in enumerate(sorted(rows, key=lambda value: int(item_ids[value]))):
                codes_by_item[int(item_ids[row])] = prefix + (collision_index,)
        return cls(item_ids.copy(), embeddings.copy(), codes_by_item, tuple(codebooks))

    def item_for_code(self, code: tuple[int, ...]) -> int | None:
        return next((item_id for item_id, candidate in self.codes.items() if candidate == code), None)

    def valid_next_tokens(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        tokens = {
            code[len(prefix)]
            for code in self.codes.values()
            if len(code) > len(prefix) and code[: len(prefix)] == prefix
        }
        return tuple(sorted(tokens))
