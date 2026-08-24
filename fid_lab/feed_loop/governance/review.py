"""Launch gates for content governance experiments.

The gates deliberately exclude hidden simulator quality and oracle utilities.
They use the same observable contract that a production randomized A/B can
reconstruct from served candidates and mature behavior logs.
"""

from __future__ import annotations

from math import ceil, isfinite

from .contracts import GovernanceLaunchThresholds


def _difference(metric):
    return metric["treatment_mean"] - metric["control_mean"]


def _power(metric, population):
    effect = _difference(metric)
    standard_error = metric["standard_error"]
    mde_80 = 2.80 * standard_error
    required = (
        population * (mde_80 / abs(effect)) ** 2
        if abs(effect) > 1e-12 else float("inf")
    )
    return {
        "observed_effect": effect,
        "standard_error": standard_error,
        "mde_80_percent_power": mde_80,
        "required_total_users_at_observed_effect": (
            ceil(required) if isfinite(required) else None
        ),
        "assumptions": "two-sided alpha=0.05, power=0.80, variance fixed",
    }


def _enabled_levers(config, trajectory_steps):
    return {
        "risk_filter": config.max_predicted_integrity_risk < 0.999,
        "diversity": (
            config.repeated_cluster_penalty > 0
            or config.repeated_author_penalty > 0
        ),
        "poi_pacing": (
            config.max_poi_per_session < trajectory_steps
            or config.min_poi_gap > 0
        ),
        "creator_exploration": config.new_creator_boost > 0,
    }


def _shadow_gates(metrics, thresholds, levers):
    gates = {
        "platform_lt_direction_nonnegative": _difference(
            metrics["lt_value_per_user"]
        ) >= 0.0,
        "platform_lt_noninferior": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= thresholds.shadow_lt_noninferiority,
        "stay_guardrail": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= thresholds.shadow_stay_noninferiority,
        "quality_view_guardrail": metrics["quality_long_view_rate"][
            "confidence_interval"
        ][0] >= thresholds.shadow_quality_view_noninferiority,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= thresholds.shadow_negative_upper,
    }
    if levers["risk_filter"]:
        gates["predicted_integrity_risk_reduced"] = metrics[
            "predicted_integrity_risk_per_exposure"
        ]["confidence_interval"][1] < 0.0
    if levers["diversity"]:
        gates["near_duplicate_noninferior"] = metrics[
            "near_duplicate_rate"
        ]["confidence_interval"][1] <= thresholds.shadow_duplicate_upper
    if levers["poi_pacing"]:
        gates["poi_load_changed"] = _difference(
            metrics["selected_poi_rate"]
        ) < 0.0
    return gates


def _online_gates(metrics, thresholds, levers):
    gates = {
        "platform_lt_direction_nonnegative": _difference(
            metrics["lt_value_per_user"]
        ) >= 0.0,
        "platform_lt_noninferior": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= thresholds.online_lt_noninferiority,
        "stay_noninferior": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= thresholds.online_stay_noninferiority,
        "quality_view_noninferior": metrics["quality_long_view_rate"][
            "confidence_interval"
        ][0] >= thresholds.online_quality_view_noninferiority,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= thresholds.online_negative_upper,
    }
    if levers["risk_filter"]:
        gates["predicted_integrity_risk_direction"] = _difference(
            metrics["predicted_integrity_risk_per_exposure"]
        ) < 0.0
    if levers["diversity"]:
        gates["near_duplicate_direction"] = _difference(
            metrics["near_duplicate_rate"]
        ) <= 0.0
    if levers["poi_pacing"]:
        gates["poi_load_direction"] = _difference(
            metrics["selected_poi_rate"]
        ) < 0.0
    return gates


def evaluate_governance_launch(
    paired_metrics, online_metrics, governance_config,
    trajectory_steps, population, thresholds=None,
):
    thresholds = thresholds or GovernanceLaunchThresholds()
    levers = _enabled_levers(governance_config, trajectory_steps)
    shadow = _shadow_gates(paired_metrics, thresholds, levers)
    online = _online_gates(online_metrics, thresholds, levers)
    if all(shadow.values()) and all(online.values()):
        decision = "pass"
    elif all(shadow.values()) and all(
        value for name, value in online.items()
        if name.endswith("_direction")
    ):
        decision = "continue_powered_online_experiment"
    else:
        decision = "hold_or_reject"
    return {
        "launch_thresholds": thresholds.manifest(),
        "enabled_levers": levers,
        "online_power": {
            name: _power(online_metrics[name], population)
            for name in (
                "lt_value_per_user", "stay_per_exposure",
                "quality_long_view_rate", "negative_rate",
            )
        },
        "shadow_gates": shadow,
        "online_gates": online,
        "decision": decision,
    }
