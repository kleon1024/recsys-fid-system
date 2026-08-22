"""Multi-route recall with a Viking-compatible vector backend boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..catalog import ItemCatalog
from ..config import RecallConfig
from ..domain import Candidate, RequestContext


@dataclass(frozen=True)
class RecallHit:
    item_id: int
    score: float
    route: str
    reason: str


class RecallRoute(Protocol):
    name: str

    def recall(self, request: RequestContext, limit: int) -> list[RecallHit]: ...


class LocalVikingIndex:
    """Exact cosine index implementing the replaceable Viking vector boundary."""

    name = "viking"

    def __init__(self, catalog: ItemCatalog, version: str = "viking-local-v1") -> None:
        self.catalog = catalog
        self.version = version
        self.item_ids = np.asarray([item.item_id for item in catalog.items])
        self.matrix = np.stack([item.embedding for item in catalog.items])

    def recall(self, request: RequestContext, limit: int) -> list[RecallHit]:
        if request.user_embedding.shape != (self.matrix.shape[1],):
            raise ValueError("query and index embedding dimensions must match")
        scores = self.matrix @ request.user_embedding
        count = min(limit, len(scores))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        return [
            RecallHit(int(self.item_ids[index]), float(scores[index]), self.name, "vector_similarity")
            for index in indices
        ]


class PopularRecall:
    name = "popular"

    def __init__(self, catalog: ItemCatalog) -> None:
        self.catalog = catalog

    def recall(self, request: RequestContext, limit: int) -> list[RecallHit]:
        del request
        items = sorted(self.catalog.items, key=lambda item: (-item.popularity, item.item_id))[:limit]
        return [RecallHit(item.item_id, item.popularity, self.name, "global_popularity") for item in items]


class FreshRecall:
    name = "fresh"

    def __init__(self, catalog: ItemCatalog) -> None:
        self.catalog = catalog

    def recall(self, request: RequestContext, limit: int) -> list[RecallHit]:
        del request
        items = sorted(self.catalog.items, key=lambda item: (item.age_hours, item.item_id))[:limit]
        return [
            RecallHit(item.item_id, 1.0 / (1.0 + item.age_hours), self.name, "item_cold_start")
            for item in items
        ]


class RecallMerger:
    def __init__(self, config: RecallConfig) -> None:
        self.config = config

    def merge(self, route_hits: dict[str, list[RecallHit]], limit: int) -> list[Candidate]:
        scores: dict[int, dict[str, float]] = {}
        reasons: dict[int, list[str]] = {}
        for route, hits in route_hits.items():
            weight = self.config.route_weights[route]
            for rank, hit in enumerate(hits, start=1):
                contribution = weight / (self.config.reciprocal_rank_constant + rank)
                scores.setdefault(hit.item_id, {})[route] = contribution
                reasons.setdefault(hit.item_id, []).append(f"{route}:{hit.reason}")
        candidates = [
            Candidate(item_id, route_scores, tuple(reasons[item_id]))
            for item_id, route_scores in scores.items()
        ]
        candidates.sort(key=lambda candidate: (-sum(candidate.recall_scores.values()), candidate.item_id))
        return candidates[:limit]
