"""Run policy changes through common-random shadow and randomized A/B gates."""

from __future__ import annotations

from dataclasses import asdict

from ...feed_loop.scale.tensor_engine import (
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)
from .catalog import policy_launches
from ..contracts import PolicyLaunchSpec


def _known_dgp_effect(control, treatment):
    report = {}
    for name in ("stay_per_exposure", "lt_rate", "hlt_rate", "negative_rate"):
        zero = control["metrics"][name]
        one = treatment["metrics"][name]
        report[name] = {
            "control": zero,
            "treatment": one,
            "relative_effect": (one - zero) / zero,
        }
    return report


def _decision(spec, ab):
    negative = ab["negative_rate"]
    hlt = ab["hlt_rate"]
    primary = ab[spec.primary_metric]
    if negative["relative_lift"] > 0.01 and negative["p_value"] < 0.05:
        return "reject_negative_guardrail"
    if hlt["relative_lift"] < -0.01 and hlt["p_value"] < 0.05:
        return "reject_hlt_guardrail"
    if primary["relative_lift"] > 0.0 and primary["p_value"] < 0.05:
        return "pass_primary_metric"
    if primary["relative_lift"] < 0.0 and primary["p_value"] < 0.05:
        return "reject_primary_regression"
    return "hold_underpowered_or_neutral"


def run_policy_launch(spec: PolicyLaunchSpec, config: TensorFeedConfig):
    control = run_tensor_feed(config, spec.control)
    treatment = run_tensor_feed(config, spec.treatment)
    ab = combine_tensor_ab(control, treatment)
    return {
        "spec": asdict(spec),
        "protocol": {
            "training": spec.training_mode,
            "shadow": "common_random_potential_worlds",
            "experiment": "stable_user_50_50",
            "gate": "primary_plus_hlt_and_negative_guardrails",
            "review": "required",
        },
        "known_dgp_effect": _known_dgp_effect(control, treatment),
        "ab": ab,
        "decision": _decision(spec, ab),
        "performance": {
            "control": control["performance"],
            "treatment": treatment["performance"],
        },
    }


def run_policy_launch_suite(config: TensorFeedConfig):
    return {
        "config": asdict(config),
        "launches": [run_policy_launch(spec, config) for spec in policy_launches()],
    }
