"""Offline/online AUC, GAUC, calibration, and slice comparison."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .contracts import PredictionRecord


@dataclass(frozen=True)
class MetricSet:
    auc: float | None
    gauc: float | None
    calibration_error: float
    records: int


@dataclass(frozen=True)
class OfflineOnlineReport:
    offline: MetricSet
    online: MetricSet
    auc_gap: float | None
    slice_auc: dict[str, tuple[float | None, float | None]]


def auc(records: list[PredictionRecord]) -> float | None:
    labels = [record.label for record in records]
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, [record.score for record in records]))


def gauc(records: list[PredictionRecord]) -> float | None:
    by_user: dict[int, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        by_user[record.user_id].append(record)
    weighted_auc = 0.0
    total = 0
    for user_records in by_user.values():
        user_auc = auc(user_records)
        if user_auc is None:
            continue
        weighted_auc += user_auc * len(user_records)
        total += len(user_records)
    return weighted_auc / total if total else None


def calibration_error(records: list[PredictionRecord], bins: int = 10) -> float:
    if not records:
        return 0.0
    scores = np.asarray([record.score for record in records])
    labels = np.asarray([record.label for record in records])
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (scores >= boundaries[index]) & (scores < boundaries[index + 1])
        if index == bins - 1:
            mask |= scores == 1.0
        if mask.any():
            error += float(mask.mean()) * abs(float(scores[mask].mean() - labels[mask].mean()))
    return error


def metric_set(records: list[PredictionRecord]) -> MetricSet:
    return MetricSet(auc(records), gauc(records), calibration_error(records), len(records))


def compare(
    offline_records: list[PredictionRecord], online_records: list[PredictionRecord]
) -> OfflineOnlineReport:
    offline = metric_set(offline_records)
    online = metric_set(online_records)
    gap = None if offline.auc is None or online.auc is None else online.auc - offline.auc
    slices = sorted({record.slice_name for record in offline_records + online_records})
    slice_auc = {
        name: (
            auc([record for record in offline_records if record.slice_name == name]),
            auc([record for record in online_records if record.slice_name == name]),
        )
        for name in slices
    }
    return OfflineOnlineReport(offline, online, gap, slice_auc)
