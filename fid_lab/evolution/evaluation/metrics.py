"""Shared model-quality metrics built on scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    ndcg_score,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray,
    scores: np.ndarray,
    bins: int = 10,
) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (scores >= boundaries[index]) & (scores < boundaries[index + 1])
        if index == bins - 1:
            mask |= scores == 1.0
        if mask.any():
            result += float(mask.mean()) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return result


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    clipped = np.clip(scores.astype(float), 1e-7, 1.0 - 1e-7)
    return {
        "auc": float(roc_auc_score(labels, clipped)),
        "pr_auc": float(average_precision_score(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped)),
        "ece": expected_calibration_error(labels, clipped),
        "ndcg": float(ndcg_score(labels.reshape(1, -1), clipped.reshape(1, -1))),
    }
