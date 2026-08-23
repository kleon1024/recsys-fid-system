"""User-level experiment inference and synthetic randomization audits."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..evolution.evaluation.ab_simulator import metric_lift
from ..value import unified_lt_launch_decision
from .contracts import Trajectory


TRAJECTORY_METRICS = {
    "exposures": "__exposures__",
    "stay_seconds": "stay_seconds",
    "stay_per_exposure": "__stay_per_exposure__",
    "plays": "plays",
    "play_3s": "play_3s",
    "slides": "slides",
    "long_views": "long_views",
    "long_view_rate": "__long_view_rate__",
    "quality_long_views": "high_quality_long_views",
    "quality_long_view_rate": "__quality_long_view_rate__",
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
    "local_value_tree_score": "local_value_tree_score",
    "lt_value": "lt_value",
    "lt_stay": "__lt_component__:stay",
    "lt_active_days": "__lt_component__:active_days",
    "lt_accepted_commercialization": "__lt_component__:accepted_commercialization",
}


def _trajectory_value(trajectory: Trajectory, attribute: str) -> float:
    exposures = max(len(trajectory.rows), 1)
    if attribute == "__exposures__":
        return float(len(trajectory.rows))
    if attribute == "__stay_per_exposure__":
        return trajectory.stay_seconds / exposures
    if attribute == "__long_view_rate__":
        return trajectory.long_views / exposures
    if attribute == "__quality_long_view_rate__":
        return trajectory.high_quality_long_views / exposures
    if attribute.startswith("__lt_component__:"):
        return float(trajectory.lt_components[attribute.split(":", 1)[1]])
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
    return unified_lt_launch_decision(metrics["lt_value"])


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
