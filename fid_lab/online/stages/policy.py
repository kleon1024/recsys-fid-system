"""Auditable ranking rules and the local policy optimizer behind the COPP boundary."""

from __future__ import annotations

from collections import Counter
from typing import Generic, TypeVar

from ..catalog import ItemCatalog
from ..config import PolicyConfig, RuleConfig
from ..domain import Candidate


C = TypeVar("C", RuleConfig, PolicyConfig)


class _ConfiguredCatalogStage(Generic[C]):
    def __init__(
        self, catalog: ItemCatalog, config: C, fresh_age_hours: float
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.fresh_age_hours = fresh_age_hours


class RankingRuleEngine(_ConfiguredCatalogStage[RuleConfig]):

    def apply(self, candidates: list[Candidate]) -> list[Candidate]:
        adjusted: list[Candidate] = []
        for candidate in candidates:
            item = self.catalog.get(candidate.item_id)
            multiplier = self.config.type_multipliers[item.content_type]
            if item.age_hours <= self.fresh_age_hours:
                multiplier *= self.config.fresh_multiplier
            if item.quality >= self.config.high_quality_threshold:
                multiplier *= self.config.quality_multiplier
            adjusted.append(candidate.update(rule_score=candidate.value_score * multiplier))
        return sorted(adjusted, key=lambda value: (-value.rule_score, value.item_id))


class ConstrainedPolicyOptimizer(_ConfiguredCatalogStage[PolicyConfig]):
    """Local implementation for the unresolved proprietary COPP adapter boundary."""

    adapter_name = "copp"
    implementation = "local-constrained-policy-v1"

    def _eligible(
        self, candidate: Candidate, creators: Counter[int], categories: Counter[str]
    ) -> bool:
        item = self.catalog.get(candidate.item_id)
        return (
            creators[item.creator_id] < self.config.max_per_creator
            and categories[item.category] < self.config.max_per_category
        )

    def select(self, candidates: list[Candidate], limit: int) -> list[Candidate]:
        fresh = [
            candidate
            for candidate in candidates
            if self.catalog.get(candidate.item_id).age_hours <= self.fresh_age_hours
        ]
        fresh.sort(key=lambda candidate: (-candidate.rule_score, candidate.item_id))
        fresh_ids = {candidate.item_id for candidate in fresh[: self.config.min_fresh]}
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.rule_score + (self.config.exploration_bonus if candidate.item_id in fresh_ids else 0.0)),
                candidate.item_id,
            ),
        )
        required = [candidate for candidate in fresh if candidate.item_id in fresh_ids]
        ordered = required + [candidate for candidate in ordered if candidate.item_id not in fresh_ids]
        creators: Counter[int] = Counter()
        categories: Counter[str] = Counter()
        selected: list[Candidate] = []
        for candidate in ordered:
            if len(selected) >= limit or not self._eligible(candidate, creators, categories):
                continue
            item = self.catalog.get(candidate.item_id)
            final_score = candidate.rule_score + (
                self.config.exploration_bonus if candidate.item_id in fresh_ids else 0.0
            )
            selected.append(candidate.update(final_score=final_score))
            creators[item.creator_id] += 1
            categories[item.category] += 1
        return sorted(selected, key=lambda value: (-value.final_score, value.item_id))
