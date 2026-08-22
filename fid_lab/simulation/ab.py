"""User-level experiment inference and synthetic randomization audits."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..evolution.evaluation.ab_simulator import metric_lift
from .contracts import Trajectory


TRAJECTORY_METRICS = {
    "exposures": "__exposures__",
    "stay_seconds": "stay_seconds",
    "stay_per_exposure": "__stay_per_exposure__",
    "plays": "plays",
    "play_3s": "play_3s",
    "slides": "slides",
    "lt_views": "long_views",
    "lt_rate": "__lt_rate__",
    "hlt_views": "high_quality_long_views",
    "hlt_rate": "__hlt_rate__",
    "likes": "likes",
    "favorites": "favorites",
    "comments": "comments",
    "shares": "shares",
    "watch_minutes": "watch_minutes",
    "anchor_impressions": "anchor_impressions",
    "anchor_clicks": "anchor_clicks",
    "poi_details": "poi_details",
    "poi_favorites": "poi_favorites",
    "orders": "orders",
    "negative_feedback": "negative_feedback",
    "sessions": "sessions",
    "returned_sessions": "returned_sessions",
    "local_service_value": "local_service_value",
    "long_term_value": "discounted_value",
}


def _trajectory_value(trajectory: Trajectory, attribute: str) -> float:
    exposures = max(len(trajectory.rows), 1)
    if attribute == "__exposures__":
        return float(len(trajectory.rows))
    if attribute == "__stay_per_exposure__":
        return trajectory.stay_seconds / exposures
    if attribute == "__lt_rate__":
        return trajectory.long_views / exposures
    if attribute == "__hlt_rate__":
        return trajectory.high_quality_long_views / exposures
    return float(getattr(trajectory, attribute))


def _bayesian_bootstrap_probability(
    control: np.ndarray, treatment: np.ndarray, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    positive = 0
    draws = 800
    for _ in range(draws):
        control_weight = rng.exponential(size=len(control))
        treatment_weight = rng.exponential(size=len(treatment))
        control_mean = float(np.average(control, weights=control_weight))
        treatment_mean = float(np.average(treatment, weights=treatment_weight))
        positive += int(treatment_mean > control_mean)
    return positive / draws


def experiment_metrics(
    control: list[Trajectory],
    treatment: list[Trajectory],
    assigned: np.ndarray,
):
    report = {}
    potential_outcomes = {}
    for name, attribute in TRAJECTORY_METRICS.items():
        zero = np.asarray(
            [_trajectory_value(value, attribute) for value in control], dtype=float
        )
        one = np.asarray(
            [_trajectory_value(value, attribute) for value in treatment], dtype=float
        )
        potential_outcomes[name] = (zero, one)
        lift = asdict(metric_lift(zero[~assigned], one[assigned], zero, one))
        true_effect = float((one - zero).mean())
        pooled_variance = float(zero.var(ddof=1) + one.var(ddof=1))
        lift["required_users_for_true_effect_80pct_power"] = (
            None
            if abs(true_effect) < 1e-12
            else int(
                np.ceil(
                    2.0
                    * pooled_variance
                    * (1.96 + 0.84) ** 2
                    / true_effect**2
                )
            )
        )
        lift["posterior_probability_positive"] = _bayesian_bootstrap_probability(
            zero[~assigned], one[assigned], 91 + len(report)
        )
        report[name] = lift
    return report, potential_outcomes


def launch_decision(metrics: dict[str, dict[str, float]]) -> str:
    negative = metrics["negative_feedback"]
    stay = metrics["stay_per_exposure"]
    hlt = metrics["hlt_rate"]
    long_term = metrics["long_term_value"]
    if negative["absolute_lift"] > 0.0 and negative["p_value"] < 0.05:
        return "reject_negative_feedback"
    if hlt["relative_lift"] is not None and hlt["relative_lift"] < -0.01:
        return "reject_hlt_guardrail" if hlt["p_value"] < 0.05 else "hold_hlt_risk"
    if (
        long_term["relative_lift"] is not None
        and long_term["relative_lift"] < -0.01
    ):
        return (
            "reject_long_term_guardrail"
            if long_term["p_value"] < 0.05
            else "hold_long_term_risk"
        )
    if stay["absolute_lift"] < 0.0 and stay["p_value"] < 0.05:
        return "reject_primary_regression"
    if stay["absolute_lift"] > 0.0 and stay["p_value"] < 0.05:
        return "pass_primary_metric"
    return "hold_underpowered_or_neutral"


def randomization_audit(potential_outcomes, seed: int, draws: int = 500):
    """Verify the user-level estimator over assignments without rerunning users."""
    rng = np.random.default_rng(seed)
    report = {}
    for name, (zero, one) in potential_outcomes.items():
        estimates = []
        users = len(zero)
        treatment_count = users // 2
        for _ in range(draws):
            treatment_index = rng.choice(users, treatment_count, replace=False)
            assigned = np.zeros(users, dtype=bool)
            assigned[treatment_index] = True
            estimates.append(float(one[assigned].mean() - zero[~assigned].mean()))
        estimates_array = np.asarray(estimates)
        true_itt = float((one - zero).mean())
        interval = np.quantile(estimates_array, (0.025, 0.975))
        report[name] = {
            "true_itt": true_itt,
            "mean_estimate": float(estimates_array.mean()),
            "estimator_bias": float(estimates_array.mean() - true_itt),
            "randomization_interval": tuple(float(value) for value in interval),
            "truth_inside_randomization_interval": bool(
                interval[0] <= true_itt <= interval[1]
            ),
        }
    return report
