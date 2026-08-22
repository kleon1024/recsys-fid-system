"""Six-route recall, RRF merge, and coarse-rank candidate truncation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Catalog, SimulationConfig


@dataclass(frozen=True)
class CandidateBatch:
    item_ids: np.ndarray
    routes: tuple[tuple[str, ...], ...]
    recall_count: int
    coarse_count: int


class CascadeCandidateProvider:
    """Production-shaped candidate opportunity; the fine model never sees the full corpus."""

    ROUTE_WEIGHTS = {
        "ann": 1.00,
        "graph": 0.85,
        "geo": 0.75,
        "fresh": 0.65,
        "long_tail": 0.60,
        "popular": 0.55,
    }

    def __init__(self, config: SimulationConfig, catalog: Catalog) -> None:
        self.config = config
        self.catalog = catalog
        self.route_limit = max(config.candidates * 2, 20)
        self.merge_limit = max(config.candidates * 5, 60)

    @staticmethod
    def _top(scores: np.ndarray, limit: int) -> np.ndarray:
        count = min(limit, len(scores))
        indices = np.argpartition(-scores, count - 1)[:count]
        return indices[np.argsort(-scores[indices], kind="stable")]

    def _routes(self, state) -> dict[str, np.ndarray]:
        estimated = self.catalog.topics @ state.observed_interest
        if state.recent_item_ids:
            recent_vector = self.catalog.topics[list(state.recent_item_ids)].mean(axis=0)
            graph = self.catalog.topics @ recent_vector
        else:
            graph = estimated * 0.5 + self.catalog.popularity * 0.5
        geo = np.where(
            self.catalog.city == state.city,
            0.65 * self.catalog.quality + 0.35 * estimated,
            -1.0,
        )
        long_tail = (
            0.45 * self.catalog.quality
            + 0.35 * (1.0 - self.catalog.popularity)
            + 0.20 * estimated
        )
        scores = {
            "ann": estimated,
            "graph": graph,
            "geo": geo,
            "fresh": self.catalog.freshness,
            "long_tail": long_tail,
            "popular": self.catalog.popularity,
        }
        return {name: self._top(value, self.route_limit) for name, value in scores.items()}

    def recall(self, state) -> CandidateBatch:
        route_hits = self._routes(state)
        seen = set(state.recent_item_ids)
        merged: dict[int, float] = {}
        item_routes: dict[int, list[str]] = {}
        for route, item_ids in route_hits.items():
            for rank, item_id in enumerate(item_ids, start=1):
                item = int(item_id)
                if item in seen:
                    continue
                merged[item] = merged.get(item, 0.0) + self.ROUTE_WEIGHTS[route] / (20 + rank)
                item_routes.setdefault(item, []).append(route)
        recalled = sorted(merged, key=lambda item: (-merged[item], item))[: self.merge_limit]
        recalled_array = np.asarray(recalled, dtype=np.int64)
        estimated = self.catalog.topics[recalled_array] @ state.observed_interest
        same_city = (self.catalog.city[recalled_array] == state.city).astype(np.float32)
        coarse = (
            0.46 * estimated
            + 0.18 * self.catalog.quality[recalled_array]
            + 0.12 * self.catalog.popularity[recalled_array]
            + 0.10 * self.catalog.freshness[recalled_array]
            + 0.08 * same_city
            + 0.06 * np.asarray([merged[item] for item in recalled])
        )
        selected_index = self._top(coarse, self.config.candidates)
        selected = recalled_array[selected_index]
        return CandidateBatch(
            selected,
            tuple(tuple(item_routes[int(item)]) for item in selected),
            len(recalled),
            len(selected),
        )
