"""Shared paired A/B statistics and repeated-seed launch decisions."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np


def paired_metric(control, treatment):
    """Return a paired normal-approximation effect report."""
    delta = (treatment - control).double().cpu().numpy()
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / sqrt(len(delta)))
    control_mean = float(control.double().mean())
    treatment_mean = float(treatment.double().mean())
    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "absolute_effect": mean,
        "relative_effect": (
            None if abs(control_mean) < 1e-12 else mean / abs(control_mean)
        ),
        "standard_error": standard_error,
        "confidence_interval": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "p_value": erfc(abs(mean / max(standard_error, 1e-12)) / sqrt(2.0)),
    }


def aggregate_launch_rows(rows, primary_metric, value_metric):
    """Require identical comparisons and stable wins across repeated seeds."""
    controls = {row["control"] for row in rows}
    treatments = {row["treatment"] for row in rows}
    if len(controls) != 1 or len(treatments) != 1:
        return {
            "stage": rows[0]["stage"],
            "control": sorted(controls),
            "treatment": sorted(treatments),
            "decision": "hold_control_divergence",
            "seed_decisions": [row["decision"] for row in rows],
        }
    metrics = {}
    for name in rows[0]["metrics"]:
        effects = np.asarray([
            row["metrics"][name]["absolute_effect"] for row in rows
        ])
        metrics[name] = {
            "mean_effect": float(effects.mean()),
            "seed_std": float(effects.std(ddof=1 if len(effects) > 1 else 0)),
            "per_seed": effects.tolist(),
        }
    passed = [row["decision"] == "pass" for row in rows]
    if all(passed):
        decision = "pass_all_seeds"
    elif metrics[primary_metric]["mean_effect"] < 0 or (
        metrics[value_metric]["mean_effect"] < 0
    ):
        decision = "reject_mean_regression"
    else:
        decision = "hold_seed_instability"
    return {
        "stage": rows[0]["stage"],
        "control": rows[0]["control"],
        "treatment": rows[0]["treatment"],
        "decision": decision,
        "seed_passes": sum(passed),
        "seed_decisions": [row["decision"] for row in rows],
        "metrics": metrics,
    }
