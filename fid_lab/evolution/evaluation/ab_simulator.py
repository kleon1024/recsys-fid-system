"""Potential-outcome simulator for product, model, and strategy A/B tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
from scipy.stats import norm

from ...scale.experiment import binary_metric_mde, sample_ratio_mismatch_p_value


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    trigger_delta: float
    watch_effect: float
    click_logit_effect: float
    order_logit_effect: float
    negative_logit_effect: float
    diversity_effect: float


SCENARIOS = {
    "product": ExperimentScenario("product", 0.03, 0.05, 0.10, 0.04, 0.00, 0.00),
    "model": ExperimentScenario("model", 0.00, 0.18, 0.14, 0.10, -0.50, 0.01),
    "strategy": ExperimentScenario("strategy", 0.00, 0.12, 0.04, 0.03, -0.50, 0.04),
}


@dataclass(frozen=True)
class MetricLift:
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    true_itt: float
    standard_error: float
    confidence_interval: tuple[float, float]
    p_value: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def metric_lift(
    control: np.ndarray,
    treatment: np.ndarray,
    potential_zero: np.ndarray,
    potential_one: np.ndarray,
) -> MetricLift:
    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    lift = treatment_mean - control_mean
    standard_error = sqrt(float(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment)))
    z_score = lift / max(standard_error, 1e-12)
    interval = (lift - 1.96 * standard_error, lift + 1.96 * standard_error)
    return MetricLift(
        control_mean,
        treatment_mean,
        lift,
        lift / max(abs(control_mean), 1e-12),
        float((potential_one - potential_zero).mean()),
        standard_error,
        interval,
        float(2.0 * norm.sf(abs(z_score))),
    )


def simulate_experiment(
    scenario: ExperimentScenario,
    users: int = 200_000,
    seed: int = 20260823,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=users)
    assigned = rng.random(users) < 0.5
    trigger_uniform = rng.random(users)
    trigger_zero = trigger_uniform < _sigmoid(-2.2 + 0.45 * latent)
    trigger_one = trigger_uniform < np.clip(
        _sigmoid(-2.2 + 0.45 * latent) + scenario.trigger_delta, 0.0, 1.0
    )
    watch_noise = rng.normal(0.0, 1.0, size=users)
    watch_zero = np.maximum(0.0, 1.7 + 0.55 * latent + watch_noise) * trigger_zero
    watch_one = (
        np.maximum(0.0, 1.7 + 0.55 * latent + watch_noise + scenario.watch_effect)
        * trigger_one
    )
    click_uniform = rng.random(users)
    click_zero = click_uniform < (_sigmoid(-2.6 + 0.5 * latent) * trigger_zero)
    click_one = click_uniform < (
        _sigmoid(-2.6 + 0.5 * latent + scenario.click_logit_effect) * trigger_one
    )
    order_uniform = rng.random(users)
    order_zero = order_uniform < (_sigmoid(-5.7 + 0.6 * latent) * trigger_zero)
    order_one = order_uniform < (
        _sigmoid(-5.7 + 0.6 * latent + scenario.order_logit_effect) * trigger_one
    )
    negative_uniform = rng.random(users)
    negative_zero = negative_uniform < (_sigmoid(-4.0 - 0.2 * latent) * trigger_zero)
    negative_one = negative_uniform < (
        _sigmoid(-4.0 - 0.2 * latent + scenario.negative_logit_effect) * trigger_one
    )
    diversity_zero = np.clip(0.56 + 0.08 * latent, 0.0, 1.0) * trigger_zero
    diversity_one = np.clip(
        0.56 + 0.08 * latent + scenario.diversity_effect, 0.0, 1.0
    ) * trigger_one
    potential = {
        "watch_minutes": (watch_zero, watch_one),
        "anchor_click": (click_zero.astype(float), click_one.astype(float)),
        "order": (order_zero.astype(float), order_one.astype(float)),
        "negative_feedback": (negative_zero.astype(float), negative_one.astype(float)),
        "creator_diversity": (diversity_zero, diversity_one),
    }
    metrics = {
        name: asdict(metric_lift(zero[~assigned], one[assigned], zero, one))
        for name, (zero, one) in potential.items()
    }
    triggered_metrics = {
        name: {
            "control_mean": float(zero[(~assigned) & trigger_zero].mean()),
            "treatment_mean": float(one[assigned & trigger_one].mean()),
        }
        for name, (zero, one) in potential.items()
    }
    for values in triggered_metrics.values():
        values["absolute_lift"] = values["treatment_mean"] - values["control_mean"]
    control_count = int((~assigned).sum())
    treatment_count = int(assigned.sum())
    baseline_click = float(click_zero.mean())
    return {
        "scenario": asdict(scenario),
        "users": users,
        "assignment": {
            "control": control_count,
            "treatment": treatment_count,
            "srm_p_value": sample_ratio_mismatch_p_value(control_count, treatment_count),
        },
        "trigger": {
            "control": float(trigger_zero[~assigned].mean()),
            "treatment": float(trigger_one[assigned].mean()),
        },
        "metrics": metrics,
        "triggered_metrics": triggered_metrics,
        "triggered_metric_warning": (
            "Post-treatment triggered comparisons are diagnostic and are biased "
            "when treatment changes trigger membership."
        ),
        "anchor_click_mde": binary_metric_mde(baseline_click, min(control_count, treatment_count)),
        "truth_covered": all(
            values["confidence_interval"][0]
            <= values["true_itt"]
            <= values["confidence_interval"][1]
            for values in metrics.values()
        ),
    }


def run_scenario_suite(users: int = 200_000) -> dict[str, object]:
    reports = {
        name: simulate_experiment(scenario, users=users)
        for name, scenario in SCENARIOS.items()
    }
    return {
        "reports": reports,
        "all_truth_covered": all(report["truth_covered"] for report in reports.values()),
    }
