"""Eligibility, pre-ranking, multi-objective prediction, and value fusion."""

from __future__ import annotations

from math import exp
from typing import Mapping

import numpy as np

from ..catalog import ItemCatalog
from ..config import ValueTreeConfig
from ..domain import Candidate, RequestContext


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-max(min(value, 30.0), -30.0)))


class EligibilityFilter:
    def __init__(self, catalog: ItemCatalog) -> None:
        self.catalog = catalog

    def apply(self, request: RequestContext, candidates: list[Candidate]) -> list[Candidate]:
        return [
            candidate
            for candidate in candidates
            if (item := self.catalog.get(candidate.item_id)).is_safe
            and item.is_active
            and request.country in item.allowed_countries
            and item.item_id not in request.seen_item_ids
        ]


class CoarseRanker:
    version = "coarse-linear-v1"

    def __init__(self, catalog: ItemCatalog) -> None:
        self.catalog = catalog

    def rank(
        self, request: RequestContext, candidates: list[Candidate], limit: int
    ) -> list[Candidate]:
        ranked: list[Candidate] = []
        for candidate in candidates:
            item = self.catalog.get(candidate.item_id)
            similarity = float(item.embedding @ request.user_embedding)
            affinity = request.category_affinity.get(item.category, 0.0)
            recall = sum(candidate.recall_scores.values())
            score = 0.58 * similarity + 0.18 * affinity + 0.14 * item.popularity + 0.10 * recall
            ranked.append(candidate.update(coarse_score=score))
        return sorted(ranked, key=lambda value: (-value.coarse_score, value.item_id))[:limit]


class ValueTree:
    """Named multi-objective fusion tree with weights owned by configuration."""

    def __init__(self, config: ValueTreeConfig) -> None:
        self.config = config

    @staticmethod
    def weighted_sum(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("value-tree sibling weights must sum to one")
        return sum(values[name] * weight for name, weight in weights.items())

    def evaluate(self, predictions: Mapping[str, float]) -> tuple[float, dict[str, float]]:
        engagement = self.weighted_sum(predictions, self.config.engagement_weights)
        ecosystem = self.weighted_sum(predictions, self.config.ecosystem_weights)
        nodes = {"engagement": engagement, "ecosystem": ecosystem}
        return self.weighted_sum(nodes, self.config.root_weights), nodes


class FineRanker:
    version = "fine-multitask-v1"

    def __init__(self, catalog: ItemCatalog, value_tree: ValueTree) -> None:
        self.catalog = catalog
        self.value_tree = value_tree

    def predict(self, request: RequestContext, item_id: int) -> dict[str, float]:
        item = self.catalog.get(item_id)
        similarity = float(np.dot(item.embedding, request.user_embedding))
        affinity = request.category_affinity.get(item.category, 0.0)
        freshness = 1.0 / (1.0 + item.age_hours / 72.0)
        return {
            "p_click": sigmoid(-1.15 + 2.0 * similarity + 0.8 * affinity + 0.35 * item.popularity),
            "p_like": sigmoid(-1.55 + 1.55 * similarity + 0.9 * affinity + 0.55 * item.quality),
            "p_long_view": sigmoid(-1.35 + 1.25 * similarity + 0.65 * item.quality),
            "quality": item.quality,
            "freshness": freshness,
        }

    def rank(
        self, request: RequestContext, candidates: list[Candidate], limit: int
    ) -> list[Candidate]:
        ranked: list[Candidate] = []
        for candidate in candidates:
            predictions = self.predict(request, candidate.item_id)
            value_score, _ = self.value_tree.evaluate(predictions)
            ranked.append(candidate.update(predictions=predictions, value_score=value_score))
        return sorted(ranked, key=lambda value: (-value.value_score, value.item_id))[:limit]
