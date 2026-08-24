"""Cross-salt aggregation for unified Feed and Local Launch Reviews."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np

from ..governance.contracts import ContentGovernanceConfig
from ..governance.review import evaluate_governance_launch
from .contracts import CompositeLaunchThresholds
from .launch import _decision, _online_gates, _shadow_gates


def _effect(cell):
    return cell["treatment_mean"] - cell["control_mean"]


def _pool(cells):
    errors = np.asarray([cell["standard_error"] for cell in cells]).clip(1e-12)
    precision = errors ** -2
    effects = np.asarray([_effect(cell) for cell in cells])
    control = np.asarray([cell["control_mean"] for cell in cells])
    effect = float((effects * precision).sum() / precision.sum())
    control_mean = float((control * precision).sum() / precision.sum())
    standard_error = float(precision.sum() ** -0.5)
    return {
        "control_mean": control_mean,
        "treatment_mean": control_mean + effect,
        "relative_lift": (
            None if abs(control_mean) < 1e-12 else effect / control_mean
        ),
        "standard_error": standard_error,
        "confidence_interval": [
            effect - 1.96 * standard_error,
            effect + 1.96 * standard_error,
        ],
        "p_value": erfc(
            abs(effect / max(standard_error, 1e-12)) / sqrt(2.0)
        ),
        "per_salt_effects": effects.tolist(),
        "estimator": "inverse_variance_pooled_user_hash_ab",
    }


def _pool_group(reports, key):
    names = reports[0][key]
    if any(set(report[key]) != set(names) for report in reports[1:]):
        raise ValueError(f"{key} metrics differ across salts")
    return {
        name: _pool([report[key][name] for report in reports])
        for name in names
    }


def _comparison_identity(report):
    return report["control"], report["treatment"], report["behavior_world"]


def aggregate_composite_launches(reports, thresholds=None):
    """Pool independent assignment salts without hiding per-salt instability."""
    if len(reports) < 3:
        raise ValueError("a powered aggregate requires at least three salts")
    if any(
        report.get("schema") != "unified-feed-business-serving-launch-v3"
        for report in reports
    ):
        raise ValueError("aggregate accepts only serving launch schema v3")
    identity = _comparison_identity(reports[0])
    if any(_comparison_identity(report) != identity for report in reports[1:]):
        raise ValueError("all salts must compare identical serving policies")
    salts = [report["config"]["experiment_salt"] for report in reports]
    if len(set(salts)) != len(salts):
        raise ValueError("experiment salts must be unique")
    thresholds = thresholds or CompositeLaunchThresholds()
    shadow = _pool_group(reports, "paired_shadow_replay")
    online = _pool_group(reports, "online_cuped_ab")
    shadow_gates = _shadow_gates(shadow, thresholds)
    online_gates = _online_gates(online, thresholds)
    online_gates["primary_direction_replicated"] = sum(
        effect > 0
        for effect in online["anchor_click_rate"]["per_salt_effects"]
    ) >= 2
    online_gates["lt_direction_replicated"] = sum(
        effect >= 0
        for effect in online["lt_value_per_user"]["per_salt_effects"]
    ) >= 2
    return {
        "schema": "unified-feed-business-serving-aggregate-v1",
        "salts": salts,
        "replicates": len(reports),
        "users_per_replicate": [report["config"]["users"] for report in reports],
        "control": reports[0]["control"],
        "treatment": reports[0]["treatment"],
        "launch_thresholds": thresholds.manifest(),
        "paired_shadow_replay": shadow,
        "online_randomized_ab": online,
        "shadow_gates": shadow_gates,
        "online_gates": online_gates,
        "decision": _decision(shadow_gates, online_gates),
        "oracle_diagnostics": {
            name: online[name]
            for name in (
                "coarse_feed_oracle_recall",
                "fine_oracle_regret_per_exposure",
            )
        },
        "evidence_boundary": (
            "Independent salted randomized experiments in the synthetic V4 "
            "world. This is not production lift evidence."
        ),
    }


def aggregate_governance_launches(reports, thresholds=None):
    """Pool governance experiments while preserving lever attribution."""
    if len(reports) < 3:
        raise ValueError("a powered aggregate requires at least three salts")
    if any(
        report.get("schema") != "content-governance-launch-v1"
        for report in reports
    ):
        raise ValueError("aggregate accepts only governance launch schema v1")
    identity = _comparison_identity(reports[0])
    if any(_comparison_identity(report) != identity for report in reports[1:]):
        raise ValueError("all salts must compare identical serving policies")
    salts = [report["config"]["experiment_salt"] for report in reports]
    if len(set(salts)) != len(salts):
        raise ValueError("experiment salts must be unique")
    shadow = _pool_group(reports, "paired_shadow_replay")
    online = _pool_group(reports, "online_cuped_ab")
    governance = ContentGovernanceConfig(
        **reports[0]["treatment"]["content_governance"]
    )
    population = sum(report["config"]["users"] for report in reports)
    trajectory_steps = (
        reports[0]["config"]["steps"] - reports[0]["warmup_steps"]
    )
    review = evaluate_governance_launch(
        shadow, online, governance, trajectory_steps, population, thresholds
    )
    replication = {
        "lt_direction_replicated": sum(
            effect >= 0
            for effect in online["lt_value_per_user"]["per_salt_effects"]
        ) >= 2,
    }
    if review["enabled_levers"]["risk_filter"]:
        replication["risk_direction_replicated"] = sum(
            effect < 0 for effect in online[
                "predicted_integrity_risk_per_exposure"
            ]["per_salt_effects"]
        ) >= 2
    if review["enabled_levers"]["diversity"]:
        replication["diversity_direction_replicated"] = sum(
            effect <= 0 for effect in online[
                "near_duplicate_rate"
            ]["per_salt_effects"]
        ) >= 2
    review["online_gates"].update(replication)
    if all(review["shadow_gates"].values()) and all(
        review["online_gates"].values()
    ):
        review["decision"] = "pass"
    elif not all(replication.values()):
        review["decision"] = "hold_or_reject"
    return {
        "schema": "content-governance-aggregate-v1",
        "salts": salts,
        "replicates": len(reports),
        "users_per_replicate": [
            report["config"]["users"] for report in reports
        ],
        "control": reports[0]["control"],
        "treatment": reports[0]["treatment"],
        "paired_shadow_replay": shadow,
        "online_randomized_ab": online,
        **review,
        "evidence_boundary": (
            "Independent salted randomized experiments in the synthetic V5 "
            "world. Hidden DGP quality is excluded from launch gates; this is "
            "not production lift evidence."
        ),
    }
