"""Pre-treatment trigger state and user-level A/B estimators."""

from __future__ import annotations

from math import erfc, sqrt

import torch

from ....simulation.experimentation.assignment import assign_binary_torch
from ..graph.reporting import CELL_METRICS


def refresh_search_state(config, state, step):
    if config.search_event_rate <= 0.0:
        return
    modulus = 2**31
    event_hash = torch.remainder(
        state["user_ids"] * 1_103_515_245
        + step * 12_345 + config.seed * 503 + 97,
        modulus,
    )
    event = event_hash.float() / float(modulus) < config.search_event_rate
    topic_hash = torch.remainder(
        state["user_ids"] * 48_271 + step * 7_919 + config.seed + 17,
        config.topics,
    )
    strength_hash = torch.remainder(
        state["user_ids"] * 69_697 + step * 503 + config.seed * 17 + 29,
        10_009,
    ).float() / 10_009
    state["search_topic"] = torch.where(
        event, topic_hash.long(), state["search_topic"]
    )
    state["search_strength"] = torch.where(
        event, 0.6 + 0.4 * strength_hash, state["search_strength"]
    )
    state["search_ttl"] = torch.where(
        event,
        torch.full_like(state["search_ttl"], config.search_ttl_requests),
        state["search_ttl"],
    )


def trigger_mask(state, kind):
    if kind == "post_search":
        return state["search_ttl"] > 0
    if kind == "retarget":
        return state["retarget_item"] >= 0
    raise ValueError(f"unsupported trigger kind: {kind}")


def _combine_experiment_cells(control_cell, treatment_cell):
    report = {}
    for name in control_cell:
        control = control_cell[name]
        treatment = treatment_cell[name]
        difference = treatment["mean"] - control["mean"]
        standard_error = sqrt(
            control["variance"] / control["users"]
            + treatment["variance"] / treatment["users"]
        )
        z_score = difference / max(standard_error, 1e-12)
        report[name] = {
            "control_mean": control["mean"],
            "treatment_mean": treatment["mean"],
            "relative_lift": (
                difference / control["mean"] if abs(control["mean"]) > 1e-12 else None
            ),
            "standard_error": standard_error,
            "confidence_interval": (
                difference - 1.96 * standard_error,
                difference + 1.96 * standard_error,
            ),
            "p_value": erfc(abs(z_score) / sqrt(2.0)),
        }
    return report


def combine_tensor_ab(control_report, treatment_report):
    """Combine stable control/treatment cells from common-random worlds."""
    return _combine_experiment_cells(
        control_report["experiment_cells"]["control"],
        treatment_report["experiment_cells"]["treatment"],
    )


def combine_tensor_cuped_ab(control_report, treatment_report):
    """CUPED-adjust disjoint hash cells using an identical pre-period."""
    required = ("_all_user_metrics", "_preperiod_user_metrics")
    if any(name not in control_report for name in required):
        raise ValueError("control report does not retain CUPED user metrics")
    if any(name not in treatment_report for name in required):
        raise ValueError("treatment report does not retain CUPED user metrics")
    control_y = control_report["_all_user_metrics"].double()
    treatment_y = treatment_report["_all_user_metrics"].double()
    control_x = control_report["_preperiod_user_metrics"].double()
    treatment_x = treatment_report["_preperiod_user_metrics"].double()
    if control_y.shape != treatment_y.shape or control_x.shape != control_y.shape:
        raise ValueError("CUPED reports have incompatible user shapes")
    preperiod_max_abs_delta = float((control_x - treatment_x).abs().max())
    if preperiod_max_abs_delta > 1e-5:
        raise ValueError(
            "CUPED pre-period differs across experiment worlds: "
            f"max_abs_delta={preperiod_max_abs_delta:.9g}"
        )
    control_salt = control_report["config"].get("experiment_salt")
    treatment_salt = treatment_report["config"].get("experiment_salt")
    if control_salt != treatment_salt:
        raise ValueError("CUPED reports use different experiment salts")
    assigned = assign_binary_torch(
        torch.arange(len(control_y)), control_salt
    )
    report = {}
    for index, name in enumerate(CELL_METRICS):
        x = control_x[:, index]
        y = torch.where(assigned, treatment_y[:, index], control_y[:, index])
        x_centered = x - x.mean()
        variance = x_centered.square().mean()
        theta = (
            (x_centered * (y - y.mean())).mean() / variance
            if float(variance) > 1e-12 else torch.zeros((), dtype=y.dtype)
        )
        adjusted = y - theta * x_centered
        left, right = adjusted[~assigned], adjusted[assigned]
        difference = float(right.mean() - left.mean())
        standard_error = sqrt(
            float(left.var(unbiased=True) / len(left))
            + float(right.var(unbiased=True) / len(right))
        )
        raw_variance = float(y.var(unbiased=True))
        adjusted_variance = float(adjusted.var(unbiased=True))
        control_mean = float(control_y[~assigned, index].mean())
        report[name] = {
            "control_mean": control_mean,
            "treatment_mean": float(treatment_y[assigned, index].mean()),
            "relative_lift": (
                difference / control_mean if abs(control_mean) > 1e-12 else None
            ),
            "standard_error": standard_error,
            "confidence_interval": (
                difference - 1.96 * standard_error,
                difference + 1.96 * standard_error,
            ),
            "p_value": erfc(
                abs(difference / max(standard_error, 1e-12)) / sqrt(2.0)
            ),
            "theta": float(theta),
            "variance_reduction": (
                1.0 - adjusted_variance / raw_variance
                if raw_variance > 1e-12 else 0.0
            ),
            "estimator": "user_hash_ab_with_preperiod_cuped",
            "preperiod_max_abs_delta": preperiod_max_abs_delta,
        }
    return report


def combine_tensor_counterfactual_ab(control_report, treatment_report):
    """Compare the same hashed users across common-random policy worlds."""
    if "_paired_user_metrics" in control_report:
        return _combine_paired_users(control_report, treatment_report)
    return _combine_experiment_cells(
        control_report["experiment_cells"]["treatment"],
        treatment_report["experiment_cells"]["treatment"],
    )


def _combine_paired_users(control_report, treatment_report):
    control = control_report["_paired_user_metrics"].double()
    treatment = treatment_report["_paired_user_metrics"].double()
    if control.shape != treatment.shape:
        raise ValueError("paired counterfactual worlds must contain the same users")
    difference = treatment - control
    report = {}
    for index, name in enumerate(CELL_METRICS):
        control_mean = float(control[:, index].mean())
        treatment_mean = float(treatment[:, index].mean())
        effect = float(difference[:, index].mean())
        standard_error = float(
            difference[:, index].std(unbiased=True) / sqrt(len(difference))
        )
        z_score = effect / max(standard_error, 1e-12)
        report[name] = {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "relative_lift": effect / control_mean if abs(control_mean) > 1e-12 else None,
            "standard_error": standard_error,
            "confidence_interval": (
                effect - 1.96 * standard_error,
                effect + 1.96 * standard_error,
            ),
            "p_value": erfc(abs(z_score) / sqrt(2.0)),
            "estimator": "same_user_paired_difference",
        }
    return report


def combine_tensor_trigger_ab(control_report, treatment_report):
    control = control_report["trigger_experiment"]
    treatment = treatment_report["trigger_experiment"]
    if control["kind"] != treatment["kind"]:
        raise ValueError("trigger experiment kinds differ")
    if control["eligible_users"] != treatment["eligible_users"]:
        raise ValueError("trigger cohorts differ across common-random worlds")
    eligible = _combine_experiment_cells(
        control["cells"]["control"], treatment["cells"]["treatment"]
    )
    rate = control["eligible_rate"]
    overall_control = control_report["experiment_cells"]["control"]
    projected = {}
    for name, metric in eligible.items():
        difference = metric["treatment_mean"] - metric["control_mean"]
        projected_difference = difference * rate
        control_mean = overall_control[name]["mean"]
        projected[name] = {
            **metric,
            "control_mean": control_mean,
            "treatment_mean": control_mean + projected_difference,
            "relative_lift": (
                projected_difference / control_mean
                if abs(control_mean) > 1e-12 else None
            ),
            "standard_error": metric["standard_error"] * rate,
            "confidence_interval": tuple(
                value * rate for value in metric["confidence_interval"]
            ),
        }
    return {
        "kind": control["kind"],
        "eligible_users": control["eligible_users"],
        "eligible_rate": rate,
        "eligible_ab": eligible,
        "projected_overall_ab": projected,
    }
