"""Minimal A/B statistics for assignment integrity and sparse binary metrics."""

from __future__ import annotations

import math

from scipy.stats import chi2


def sample_ratio_mismatch_p_value(control: int, treatment: int, ratio: float = 0.5) -> float:
    total = control + treatment
    if total <= 0 or ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("counts and allocation ratio must be valid")
    expected = (total * (1.0 - ratio), total * ratio)
    statistic = (control - expected[0]) ** 2 / expected[0]
    statistic += (treatment - expected[1]) ** 2 / expected[1]
    return float(chi2.sf(statistic, df=1))


def binary_metric_mde(
    baseline_rate: float,
    users_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
    cluster_inflation: float = 1.0,
) -> float:
    if not 0.0 < baseline_rate < 1.0 or users_per_arm <= 0:
        raise ValueError("baseline rate and users per arm must be valid")
    if alpha != 0.05 or power != 0.8:
        raise ValueError("reference implementation supports alpha=0.05 and power=0.8")
    standard_error = math.sqrt(
        2.0 * baseline_rate * (1.0 - baseline_rate) * cluster_inflation / users_per_arm
    )
    return 2.8016 * standard_error
