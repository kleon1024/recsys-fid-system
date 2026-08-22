"""Vectorized experiment-power layer for industrial per-mille Feed effects.

This module does not replace the stateful trajectory simulator. It answers a
different question: once a trajectory experiment estimates a plausible effect
and variance, can a million-user A/B recover a 0.1%-1% ITT without mistaking
sampling noise for product impact?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt

import numpy as np


@dataclass(frozen=True)
class SmallEffectResult:
    users: int
    target_relative_effect: float
    realized_true_relative_effect: float
    observed_relative_lift: float
    standard_error: float
    p_value: float
    confidence_interval: tuple[float, float]
    cuped_relative_lift: float
    cuped_standard_error: float
    cuped_p_value: float
    cuped_confidence_interval: tuple[float, float]
    variance_reduction: float
    truth_inside_cuped_interval: bool


def _two_sided_p(z_score: float) -> float:
    return float(erfc(abs(z_score) / sqrt(2.0)))


def _estimate(control: np.ndarray, treatment: np.ndarray):
    difference = float(treatment.mean() - control.mean())
    standard_error = float(
        sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    )
    interval = (
        difference - 1.96 * standard_error,
        difference + 1.96 * standard_error,
    )
    p_value = _two_sided_p(difference / max(standard_error, 1e-12))
    return difference, standard_error, p_value, interval


def run_small_effect_ab(
    users: int = 1_000_000,
    relative_effects: tuple[float, ...] = (0.001, 0.003, 0.005, 0.01),
    baseline_mean: float = 120.0,
    baseline_standard_deviation: float = 85.0,
    prepost_correlation: float = 0.65,
    seed: int = 20260823,
) -> list[dict[str, object]]:
    """Run one shared million-user population through several known ITTs."""
    if users < 1_000 or users % 2:
        raise ValueError("users must be an even number >= 1,000")
    rng = np.random.default_rng(seed)
    stable_user = rng.normal(size=users)
    pre_noise = rng.normal(size=users)
    post_noise = rng.normal(size=users)
    # sqrt(rho) on the shared component makes Corr(pre, post) equal rho.
    shared_weight = sqrt(prepost_correlation)
    noise_weight = sqrt(1.0 - prepost_correlation)
    pre = baseline_mean + baseline_standard_deviation * (
        shared_weight * stable_user + noise_weight * pre_noise
    )
    control_potential = baseline_mean + baseline_standard_deviation * (
        shared_weight * stable_user + noise_weight * post_noise
    )
    assigned = np.zeros(users, dtype=bool)
    assigned[rng.choice(users, users // 2, replace=False)] = True
    reports = []
    for effect in relative_effects:
        treatment_potential = control_potential + baseline_mean * effect
        observed_control = control_potential[~assigned]
        observed_treatment = treatment_potential[assigned]
        difference, standard_error, p_value, interval = _estimate(
            observed_control, observed_treatment
        )

        observed_post = np.where(assigned, treatment_potential, control_potential)
        theta = float(np.cov(observed_post, pre, ddof=1)[0, 1] / pre.var(ddof=1))
        adjusted = observed_post - theta * (pre - pre.mean())
        adjusted_difference, adjusted_se, adjusted_p, adjusted_interval = _estimate(
            adjusted[~assigned], adjusted[assigned]
        )
        true_difference = float((treatment_potential - control_potential).mean())
        report = SmallEffectResult(
            users=users,
            target_relative_effect=effect,
            realized_true_relative_effect=true_difference / control_potential.mean(),
            observed_relative_lift=difference / observed_control.mean(),
            standard_error=standard_error,
            p_value=p_value,
            confidence_interval=interval,
            cuped_relative_lift=adjusted_difference / adjusted[~assigned].mean(),
            cuped_standard_error=adjusted_se,
            cuped_p_value=adjusted_p,
            cuped_confidence_interval=adjusted_interval,
            variance_reduction=1.0 - adjusted_se**2 / standard_error**2,
            truth_inside_cuped_interval=bool(
                adjusted_interval[0] <= true_difference <= adjusted_interval[1]
            ),
        )
        reports.append(asdict(report))
    return reports
