"""Launch Review for the unified Feed and Local serving graph."""

from __future__ import annotations

from dataclasses import asdict

from ..scale.experiment.trigger import (
    combine_tensor_ab,
    combine_tensor_counterfactual_ab,
    combine_tensor_cuped_ab,
)
from ..scale.tensor_engine import run_tensor_feed
from .composite import CompositeTensorPolicy
from .contracts import CompositeLaunchThresholds


def _shadow_gates(metrics, thresholds):
    return {
        "local_primary_positive": metrics["anchor_click_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_direction_nonnegative": (
            metrics["lt_value_per_user"]["treatment_mean"]
            >= metrics["lt_value_per_user"]["control_mean"]
        ),
        "platform_lt_noninferior": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= thresholds.shadow_lt_noninferiority,
        "stay_guardrail": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= thresholds.shadow_stay_noninferiority,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= thresholds.shadow_negative_upper,
        "coarse_recall_noninferior": metrics["coarse_feed_oracle_recall"][
            "confidence_interval"
        ][0] >= thresholds.shadow_coarse_recall_noninferiority,
    }


def _online_gates(metrics, thresholds):
    """Gate only on metrics observable in a production randomized A/B.

    Simulator oracle recall remains a shadow diagnostic. Treating it as an
    online gate would make the synthetic world leak into the experiment
    authority and could not be reproduced in production.
    """
    return {
        "local_primary_positive": metrics["anchor_click_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_direction_nonnegative": (
            metrics["lt_value_per_user"]["treatment_mean"]
            >= metrics["lt_value_per_user"]["control_mean"]
        ),
        "platform_lt_noninferior": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= thresholds.online_lt_noninferiority,
        "stay_noninferior": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= thresholds.online_stay_noninferiority,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= thresholds.online_negative_upper,
    }


def _decision(shadow_gates, online_gates):
    if all(shadow_gates.values()) and all(online_gates.values()):
        return "pass"
    if all(shadow_gates.values()) and online_gates["local_primary_positive"]:
        return "continue_powered_online_experiment"
    return "hold_or_reject"


def _control_policy(feed_policy, control_local_bundle, value_config):
    return (
        feed_policy if control_local_bundle is None
        else CompositeTensorPolicy(
            feed_policy, control_local_bundle, value_config
        )
    )


def run_composite_serving_launch(
    config, feed_policy, local_bundle, behavior_world, value_config=None,
    control_local_bundle=None, treatment_coarse_local_bundle=None,
    warmup_steps=0, thresholds=None,
):
    thresholds = thresholds or CompositeLaunchThresholds()
    treatment = CompositeTensorPolicy(
        feed_policy, local_bundle, value_config, treatment_coarse_local_bundle
    )
    control = _control_policy(
        feed_policy, control_local_bundle, value_config
    )
    if not 0 <= warmup_steps < config.steps:
        raise ValueError("composite warmup must precede the measurement window")
    control_schedule = tuple(control for _ in range(config.steps))
    treatment_schedule = (
        tuple(control for _ in range(warmup_steps))
        + tuple(treatment for _ in range(config.steps - warmup_steps))
    )
    control_world = run_tensor_feed(
        config, control, policy_schedule=control_schedule,
        measurement_start_step=warmup_steps, behavior_world=behavior_world,
    )
    treatment_world = run_tensor_feed(
        config, treatment, policy_schedule=treatment_schedule,
        measurement_start_step=warmup_steps, behavior_world=behavior_world,
    )
    paired = combine_tensor_counterfactual_ab(
        control_world, treatment_world
    )
    online = combine_tensor_ab(control_world, treatment_world)
    cuped = combine_tensor_cuped_ab(control_world, treatment_world)
    shadow_gates = _shadow_gates(paired, thresholds)
    online_gates = _online_gates(cuped, thresholds)
    return {
        "schema": "unified-feed-business-serving-launch-v3",
        "config": asdict(config),
        "control": control.describe(),
        "treatment": treatment.describe(),
        "paired_shadow_replay": paired,
        "online_disjoint_ab": online,
        "online_cuped_ab": cuped,
        "warmup_steps": warmup_steps,
        "launch_thresholds": thresholds.manifest(),
        "shadow_gates": shadow_gates,
        "online_gates": online_gates,
        "online_diagnostics": {
            "coarse_feed_oracle_recall": cuped["coarse_feed_oracle_recall"],
            "fine_oracle_regret_per_exposure": cuped[
                "fine_oracle_regret_per_exposure"
            ],
        },
        "decision": _decision(shadow_gates, online_gates),
        "candidate_graph": treatment_world["candidate_graph"],
        "performance": {
            "control": control_world["performance"],
            "treatment": treatment_world["performance"],
        },
        "behavior_world": treatment_world["behavior_world"],
        "evidence_boundary": (
            "Unified main-Feed and Local model composition in the external V4 "
            "simulator. This is synthetic Launch Review evidence only."
        ),
    }
