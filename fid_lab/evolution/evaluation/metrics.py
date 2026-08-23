"""Shared model-quality metrics built on scikit-learn."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    ndcg_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class GroupedAUC:
    """GAUC plus the coverage hidden by a single scalar."""

    value: float | None
    total_groups: int
    eligible_groups: int
    eligible_group_rate: float
    total_records: int
    eligible_records: int
    eligible_record_rate: float


def grouped_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float | int | None]:
    """Record-weighted within-group AUC; single-class groups are reported, not hidden."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    group_ids = np.asarray(group_ids)
    if not (len(labels) == len(scores) == len(group_ids)):
        raise ValueError("labels, scores, and group_ids must have equal length")
    auc_weight = 0.0
    eligible_records = 0
    eligible_groups = 0
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    slices = np.split(order, boundaries)
    for indices in slices:
        group_labels = labels[indices]
        if np.unique(group_labels).size < 2:
            continue
        records = len(indices)
        auc_weight += float(roc_auc_score(group_labels, scores[indices])) * records
        eligible_records += records
        eligible_groups += 1
    total_records = len(labels)
    total_groups = len(slices)
    result = GroupedAUC(
        value=auc_weight / eligible_records if eligible_records else None,
        total_groups=total_groups,
        eligible_groups=eligible_groups,
        eligible_group_rate=eligible_groups / total_groups if total_groups else 0.0,
        total_records=total_records,
        eligible_records=eligible_records,
        eligible_record_rate=eligible_records / total_records if total_records else 0.0,
    )
    return asdict(result)


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
