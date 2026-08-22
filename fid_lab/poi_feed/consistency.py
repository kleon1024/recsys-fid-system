"""Full-path and cascade consistency checks for the POI Feed vertical."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import PoiFeedExample


@dataclass(frozen=True)
class FullPathAudit:
    passed: bool
    checks: dict[str, bool]
    coarse_positive_recall: float
    fine_positive_recall: float
    max_feature_delta: float


class FullPathConsistencyAuditor:
    def audit(
        self,
        examples: tuple[PoiFeedExample, ...],
        expected_versions: dict[str, str],
        offline_features: np.ndarray,
        online_features: np.ndarray,
        positive_candidate_ids: set[int],
        coarse_candidate_ids: set[int],
        fine_candidate_ids: set[int],
        tolerance: float = 1e-7,
    ) -> FullPathAudit:
        if offline_features.shape != online_features.shape:
            raise ValueError("offline and online feature shapes must match")
        delta = float(np.max(np.abs(offline_features - online_features)))
        denominator = max(len(positive_candidate_ids), 1)
        coarse_recall = len(positive_candidate_ids & coarse_candidate_ids) / denominator
        fine_recall = len(positive_candidate_ids & fine_candidate_ids) / denominator
        checks = {
            "media_version": all(
                value.media_version == expected_versions["media"] for value in examples
            ),
            "feature_version": all(
                value.feature_version == expected_versions["feature"] for value in examples
            ),
            "model_version": all(
                value.model_version == expected_versions["model"] for value in examples
            ),
            "index_version": all(
                value.index_version == expected_versions["index"] for value in examples
            ),
            "feature_parity": delta <= tolerance,
            "coarse_pass_through": coarse_recall >= 0.95,
            "fine_pass_through": fine_recall >= 0.90,
            "point_in_time_sequence": all(
                value.sequence.shape == (24, 8) for value in examples
            ),
        }
        return FullPathAudit(all(checks.values()), checks, coarse_recall, fine_recall, delta)
