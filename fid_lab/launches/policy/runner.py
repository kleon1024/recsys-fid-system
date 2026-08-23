"""Run policy changes through common-random shadow and randomized A/B gates."""

from __future__ import annotations

from dataclasses import asdict

from ...value import unified_lt_launch_decision
from ...feed_loop.scale.tensor_engine import (
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)
from .catalog import policy_launches
from ..contracts import PolicyLaunchSpec


def _known_dgp_effect(control, treatment):
    report = {}
    for name in (
        "stay_per_exposure",
        "long_view_rate",
        "quality_long_view_rate",
        "negative_rate",
        "lt_value_per_exposure",
        "local_value_tree_score_per_exposure",
    ):
        zero = control["metrics"][name]
        one = treatment["metrics"][name]
        report[name] = {
            "control": zero,
            "treatment": one,
            "relative_effect": (one - zero) / zero,
        }
    return report


def _decision(spec, ab):
    del spec
    return unified_lt_launch_decision(ab["lt_value_per_user"])


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
            "gate": "unified_exchanged_lt_with_independent_hard_constraints",
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
