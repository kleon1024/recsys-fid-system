"""Final cross-content mixing, calibration, quotas, pinning, and diversity."""

from __future__ import annotations

from collections import Counter

from ..catalog import ItemCatalog
from ..config import MixConfig
from ..domain import Candidate, RequestContext


class MixedRanker:
    def __init__(self, catalog: ItemCatalog, config: MixConfig) -> None:
        self.catalog = catalog
        self.config = config

    def calibrated(self, candidate: Candidate) -> float:
        item_type = self.catalog.get(candidate.item_id).content_type
        scale, bias = self.config.score_calibration[item_type]
        return scale * candidate.final_score + bias

    def _breaks_diversity(self, candidate: Candidate, selected: list[Candidate]) -> bool:
        window = self.config.max_consecutive_category
        if len(selected) < window:
            return False
        category = self.catalog.get(candidate.item_id).category
        return all(self.catalog.get(value.item_id).category == category for value in selected[-window:])

    def mix(self, request: RequestContext, candidates: list[Candidate]) -> list[Candidate]:
        ordered = sorted(candidates, key=lambda value: (-self.calibrated(value), value.item_id))
        counts: Counter[str] = Counter()
        selected: list[Candidate] = []
        deferred: list[Candidate] = []
        for candidate in ordered:
            item = self.catalog.get(candidate.item_id)
            if counts[item.content_type] >= self.config.max_by_type[item.content_type]:
                continue
            if self._breaks_diversity(candidate, selected):
                deferred.append(candidate)
                continue
            selected.append(candidate.update(final_score=self.calibrated(candidate)))
            counts[item.content_type] += 1
            if len(selected) >= request.size:
                break
        for candidate in deferred:
            if len(selected) >= request.size:
                break
            item = self.catalog.get(candidate.item_id)
            if counts[item.content_type] < self.config.max_by_type[item.content_type]:
                selected.append(candidate.update(final_score=self.calibrated(candidate)))
                counts[item.content_type] += 1
        if request.pinned_item_id is not None:
            pinned = next((value for value in selected if value.item_id == request.pinned_item_id), None)
            if pinned is not None:
                selected.remove(pinned)
                selected.insert(0, pinned)
        return selected[: request.size]
