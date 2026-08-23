"""Six-route recall, RRF merge, and coarse-rank candidate truncation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .contracts import Catalog, SimulationConfig

if TYPE_CHECKING:
    from ..evolution.models.retrieval import RetrievalSnapshot
    from .experimentation.contracts import FeedParameters


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
        "post_search": 0.95,
        "retarget": 0.90,
    }

    def __init__(
        self,
        config: SimulationConfig,
        catalog: Catalog,
        parameters: FeedParameters | None = None,
        retrieval_snapshot: RetrievalSnapshot | None = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.parameters = parameters
        self.retrieval_snapshot = retrieval_snapshot
        self.route_limit = max(config.candidates * 2, 20)
        self.merge_limit = (
            parameters.recall_budget
            if parameters is not None
            else max(config.candidates * 5, 60)
        )
        if parameters is not None and parameters.coarse_budget != config.candidates:
            raise ValueError("Feed parameter coarse_budget must match simulator candidates")

    @staticmethod
    def _top(scores: np.ndarray, limit: int) -> np.ndarray:
        count = min(limit, len(scores))
        indices = np.argpartition(-scores, count - 1)[:count]
        return indices[np.argsort(-scores[indices], kind="stable")]

    def _routes(self, state) -> dict[str, np.ndarray]:
        estimated = self.catalog.topics @ state.observed_interest
        recall_model = self.parameters.recall_model if self.parameters else "two_tower_v1"
        if recall_model == "two_tower_trained_v2":
            if self.retrieval_snapshot is None:
                raise ValueError("trained retrieval model requires a retrieval snapshot")
            if len(self.retrieval_snapshot.item_embeddings) != len(self.catalog.topics):
                raise ValueError("retrieval snapshot corpus does not match serving catalog")
            ann_score = self.retrieval_snapshot.scores(state.observed_interest)
        elif recall_model == "two_tower_v1":
            ann_score = estimated
        else:
            raise ValueError(f"unsupported recall model: {recall_model}")
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
        search = np.where(
            self.catalog.category == state.search_topic,
            0.55 * estimated
            + 0.25 * self.catalog.poi_quality
            + 0.20 * (self.catalog.city == state.city),
            -1.0,
        )
        if state.recent_poi_ids:
            retarget = np.where(
                np.isin(self.catalog.poi, np.asarray(state.recent_poi_ids)),
                0.55 * self.catalog.poi_quality + 0.45 * estimated,
                -1.0,
            )
        else:
            retarget = np.where(
                self.catalog.is_poi_video & (self.catalog.city == state.city),
                0.6 * self.catalog.poi_quality + 0.4 * estimated,
                -1.0,
            )
        scores = {
            "ann": ann_score,
            "graph": graph,
            "geo": geo,
            "fresh": self.catalog.freshness,
            "long_tail": long_tail,
            "popular": self.catalog.popularity,
            "post_search": search,
            "retarget": retarget,
        }
        enabled = set(self.parameters.enabled_routes) if self.parameters else set(scores)
        unknown = enabled - set(scores)
        if unknown:
            raise ValueError(f"unsupported recall routes: {sorted(unknown)}")
        return {
            name: self._top(value, self.route_limit)
            for name, value in scores.items()
            if name in enabled
        }

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
        quality = self.catalog.quality[recalled_array]
        popularity = self.catalog.popularity[recalled_array]
        freshness = self.catalog.freshness[recalled_array]
        poi_quality = self.catalog.poi_quality[recalled_array]
        route_score = np.asarray([merged[item] for item in recalled])
        model = self.parameters.coarse_model if self.parameters else "lr_v1"
        if model == "quality_only":
            coarse = quality
        elif model == "dcnv2_distilled":
            coarse = (
                0.38 * estimated
                + 0.18 * quality
                + 0.10 * popularity
                + 0.10 * freshness
                + 0.08 * same_city
                + 0.04 * route_score
                + 0.02 * poi_quality
                + 0.07 * estimated * quality
                + 0.03 * same_city * freshness
            )
        elif model == "lr_v1":
            coarse = (
                0.46 * estimated
                + 0.18 * quality
                + 0.12 * popularity
                + 0.10 * freshness
                + 0.08 * same_city
                + 0.04 * route_score
                + 0.02 * poi_quality
            )
        else:
            raise ValueError(f"unsupported coarse model: {model}")
        selected_index = self._top(coarse, self.config.candidates)
        selected = recalled_array[selected_index]
        return CandidateBatch(
            selected,
            tuple(tuple(item_routes[int(item)]) for item in selected),
            len(recalled),
            len(selected),
        )
