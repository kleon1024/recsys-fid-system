"""Pre-treatment trigger state and user-level A/B estimators."""

from __future__ import annotations

from math import erfc, sqrt

import torch


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
