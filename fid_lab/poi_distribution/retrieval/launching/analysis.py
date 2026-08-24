"""Pooled A/B analysis for isolated ANN-route retrieval launches."""

from __future__ import annotations

import numpy as np


PRIMARY = "lt_value_per_user"


def _pooled(rows, metric):
    effects = np.asarray([
        row["metrics"][metric]["treatment_mean"]
        - row["metrics"][metric]["control_mean"]
        for row in rows
    ])
    errors = np.asarray([
        row["metrics"][metric]["standard_error"] for row in rows
    ]).clip(1e-12)
    precision = errors ** -2
    effect = float((effects * precision).sum() / precision.sum())
    error = float(precision.sum() ** -0.5)
    return {
        "per_seed": effects.tolist(),
        "pooled_effect": effect,
        "pooled_standard_error": error,
        "pooled_confidence_interval": [effect - 1.96 * error, effect + 1.96 * error],
    }


def aggregate(index, seed_reports, offline_control, offline_treatment):
    rows = [report["comparisons"][index] for report in seed_reports]
    metric_names = (
        PRIMARY, "stay_per_exposure", "negative_rate", "anchor_click_rate",
        "poi_candidate_fraction", "conversion_rate",
    )
    metrics = {name: _pooled(rows, name) for name in metric_names}
    gates = {
        "offline_recall_positive": (
            offline_treatment["recall_at_k"] > offline_control["recall_at_k"]
        ),
        "platform_lt_positive": metrics[PRIMARY]["pooled_confidence_interval"][0] > 0,
        "stay_guardrail": (
            metrics["stay_per_exposure"]["pooled_confidence_interval"][0] >= -0.02
        ),
        "negative_guardrail": (
            metrics["negative_rate"]["pooled_confidence_interval"][1] <= 0.000002
        ),
        "anchor_guardrail": (
            metrics["anchor_click_rate"]["pooled_confidence_interval"][0] >= -0.00002
        ),
        "conversion_guardrail": (
            metrics["conversion_rate"]["pooled_confidence_interval"][0] >= -0.000005
        ),
        "direction_consistency": (
            sum(value > 0 for value in metrics[PRIMARY]["per_seed"]) >= 2
        ),
    }
    if all(gates.values()):
        decision = "pass_pooled_three_seeds"
    elif (
        metrics[PRIMARY]["pooled_effect"] < 0
    ):
        decision = "reject_mean_regression"
    else:
        decision = "hold_uncertainty"
    return {
        "stage": "retrieval",
        "control": rows[0]["control"],
        "treatment": rows[0]["treatment"],
        "decision": decision,
        "gates": gates,
        "offline_control": offline_control,
        "offline_treatment": offline_treatment,
        "metrics": metrics,
        "online_disjoint_by_seed": [
            {
                name: row["online_disjoint_metrics"][name]
                for name in (PRIMARY, "anchor_click_rate")
            }
            for row in rows
        ],
        "seed_diagnostics": [
            {
                "candidate_graph": row["candidate_graph"],
                "performance": row["performance"],
            }
            for row in rows
        ],
    }
