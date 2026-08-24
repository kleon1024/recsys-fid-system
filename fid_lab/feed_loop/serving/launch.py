"""Launch Review for the unified Feed and Local serving graph."""

from __future__ import annotations

from dataclasses import asdict

from ..scale.experiment.trigger import (
    combine_tensor_ab,
    combine_tensor_counterfactual_ab,
)
from ..scale.tensor_engine import run_tensor_feed
from .composite import CompositeTensorPolicy


def _shadow_gates(metrics):
    return {
        "local_primary_positive": metrics["anchor_click_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_nonnegative": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= 0,
        "stay_guardrail": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= -0.02,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= 0.0002,
        "coarse_recall_nonnegative": metrics["coarse_feed_oracle_recall"][
            "confidence_interval"
        ][0] >= 0,
    }


def _online_gates(metrics):
    return {
        "local_primary_positive": metrics["anchor_click_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_positive": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= 0,
        "stay_nonnegative": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= 0,
        "negative_nonpositive": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= 0,
    }


def _decision(shadow_gates, online_gates):
    if all(shadow_gates.values()) and all(online_gates.values()):
        return "pass"
    if all(shadow_gates.values()) and online_gates["local_primary_positive"]:
        return "continue_powered_online_experiment"
    return "hold_or_reject"


def run_composite_serving_launch(
    config, feed_policy, local_bundle, behavior_world, value_config=None,
):
    treatment = CompositeTensorPolicy(
        feed_policy, local_bundle, value_config
    )
    control_world = run_tensor_feed(
        config, feed_policy, behavior_world=behavior_world
    )
    treatment_world = run_tensor_feed(
        config, treatment, behavior_world=behavior_world
    )
    paired = combine_tensor_counterfactual_ab(
        control_world, treatment_world
    )
    online = combine_tensor_ab(control_world, treatment_world)
    shadow_gates = _shadow_gates(paired)
    online_gates = _online_gates(online)
    return {
        "schema": "unified-feed-business-serving-launch-v1",
        "config": asdict(config),
        "control": feed_policy.describe(),
        "treatment": treatment.describe(),
        "paired_shadow_replay": paired,
        "online_disjoint_ab": online,
        "shadow_gates": shadow_gates,
        "online_gates": online_gates,
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
