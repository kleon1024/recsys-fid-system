"""Manifest, feature replay, and prediction-shadow consistency checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ChainManifest, TrainingExample


@dataclass(frozen=True)
class ConsistencyReport:
    passed: bool
    checks: dict[str, bool]
    max_prediction_delta: float


class ChainConsistencyAuditor:
    def audit(
        self,
        expected: ChainManifest,
        served: ChainManifest,
        example: TrainingExample,
        online_fids: tuple[int, ...],
        offline_scores: np.ndarray,
        online_scores: np.ndarray,
        tolerance: float = 1e-8,
    ) -> ConsistencyReport:
        if offline_scores.shape != online_scores.shape:
            raise ValueError("shadow score shapes must match")
        delta = float(np.max(np.abs(offline_scores - online_scores)))
        checks = {
            "schema_version": expected.schema_version == served.schema_version,
            "fid_layout": expected.fid_layout == served.fid_layout,
            "joiner_version": expected.joiner_version == served.joiner_version,
            "model_version": expected.model_version == served.model_version,
            "vector_index_version": expected.vector_index_version == served.vector_index_version,
            "task_contract": expected.tasks == served.tasks,
            "feature_replay": example.feature_fids == online_fids,
            "prediction_shadow": delta <= tolerance,
        }
        return ConsistencyReport(all(checks.values()), checks, delta)
