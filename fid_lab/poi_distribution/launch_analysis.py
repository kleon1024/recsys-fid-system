"""Cross-seed aggregation and release gates for POI distribution launches."""

from __future__ import annotations

import numpy as np


PRIMARY_BY_STAGE = {
    "coarse": "coarse_feed_oracle_recall",
    "fine": "anchor_click_rate",
    "mix": "local_value_tree_score_per_exposure",
    "end_to_end": "anchor_click_rate",
}


def gate(metrics, stage):
    primary = PRIMARY_BY_STAGE[stage]
    gates = {
        "stage_primary_positive": metrics[primary]["confidence_interval"][0] > 0,
        "platform_lt_nonnegative": (
            metrics["lt_value_per_user"]["confidence_interval"][0] >= 0
        ),
        "stay_guardrail": (
            metrics["stay_per_exposure"]["confidence_interval"][0] >= -0.02
        ),
        "negative_guardrail": (
            metrics["negative_rate"]["confidence_interval"][1] <= 0.0002
        ),
        "anchor_guardrail": (
            metrics["anchor_click_rate"]["confidence_interval"][0] >= -0.0002
            if stage == "coarse"
            else True
        ),
    }
    return gates, "pass" if all(gates.values()) else "hold_or_reject"


def aggregate(stage, index, reports):
    rows = [report["stages"][stage][index] for report in reports]
    metrics = {}
    for metric in rows[0]["metrics"]:
        effects = np.asarray([
            row["metrics"][metric]["treatment_mean"]
            - row["metrics"][metric]["control_mean"]
            for row in rows
        ])
        standard_errors = np.asarray([
            row["metrics"][metric]["standard_error"] for row in rows
        ]).clip(1e-12)
        inverse_variance = standard_errors ** -2
        pooled = float((effects * inverse_variance).sum() / inverse_variance.sum())
        pooled_standard_error = float(inverse_variance.sum() ** -0.5)
        metrics[metric] = {
            "mean_effect": float(effects.mean()),
            "seed_std": float(effects.std(ddof=1)),
            "per_seed": effects.tolist(),
            "pooled_effect": pooled,
            "pooled_standard_error": pooled_standard_error,
            "pooled_confidence_interval": [
                pooled - 1.96 * pooled_standard_error,
                pooled + 1.96 * pooled_standard_error,
            ],
        }
    primary = PRIMARY_BY_STAGE[stage]
    pooled_gates = {
        "stage_primary_positive": metrics[primary]["pooled_confidence_interval"][0] > 0,
        "platform_lt_nonnegative": (
            metrics["lt_value_per_user"]["pooled_confidence_interval"][0] >= 0
        ),
        "stay_guardrail": (
            metrics["stay_per_exposure"]["pooled_confidence_interval"][0] >= -0.02
        ),
        "negative_guardrail": (
            metrics["negative_rate"]["pooled_confidence_interval"][1] <= 0.0002
        ),
        "direction_consistency": (
            sum(value > 0 for value in metrics[primary]["per_seed"]) >= 2
            and sum(value >= 0 for value in metrics["lt_value_per_user"]["per_seed"]) >= 2
        ),
        "anchor_guardrail": (
            metrics["anchor_click_rate"]["pooled_confidence_interval"][0] >= -0.0002
            if stage == "coarse"
            else True
        ),
    }
    if all(pooled_gates.values()):
        decision = "pass_pooled_three_seeds"
    elif metrics[primary]["mean_effect"] < 0 or metrics["lt_value_per_user"]["mean_effect"] < 0:
        decision = "reject_mean_regression"
    else:
        decision = "hold_seed_instability"
    return {
        "stage": stage,
        "control": rows[0]["control"],
        "treatment": rows[0]["treatment"],
        "decision": decision,
        "seed_passes": sum(row["decision"] == "pass" for row in rows),
        "pooled_gates": pooled_gates,
        "metrics": metrics,
        "seed_reports": rows,
    }
