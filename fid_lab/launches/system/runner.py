"""Architecture and measurement-bug launches using the same A/B evidence shape."""

from __future__ import annotations

from dataclasses import asdict, replace

from ...feed_loop.scale.tensor_engine import (
    PERSONALIZED_1PCT,
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)


def _distribution_delta(control, treatment):
    names = (
        "stay_per_exposure",
        "lt_rate",
        "hlt_rate",
        "negative_rate",
        "play_rate",
    )
    return {
        name: (treatment["metrics"][name] - control["metrics"][name])
        / control["metrics"][name]
        for name in names
    }


def run_architecture_launch(config: TensorFeedConfig):
    control_config = replace(config, batch_users=10_000)
    treatment_config = replace(config, batch_users=25_000)
    control = run_tensor_feed(control_config, PERSONALIZED_1PCT)
    treatment = run_tensor_feed(treatment_config, PERSONALIZED_1PCT)
    ab = combine_tensor_ab(control, treatment)
    delta = _distribution_delta(control, treatment)
    throughput_lift = (
        treatment["performance"]["requests_per_second"]
        / control["performance"]["requests_per_second"]
        - 1.0
    )
    parity = max(abs(delta[name]) for name in delta if name != "negative_rate") < 0.01
    decision = (
        "pass_parity_and_performance"
        if parity and throughput_lift > 0.0
        else "reject_parity_or_performance"
    )
    return {
        "launch_id": "L-ARCH-001",
        "category": "architecture",
        "title": "Increase GPU user batch",
        "training": "not_applicable_runtime_only",
        "control_config": asdict(control_config),
        "treatment_config": asdict(treatment_config),
        "distribution_relative_delta": delta,
        "ab": ab,
        "throughput_lift": throughput_lift,
        "control_performance": control["performance"],
        "treatment_performance": treatment["performance"],
        "decision": decision,
    }


def run_bug_fix_launch(config: TensorFeedConfig):
    bug_config = replace(config, count_inactive_play_bug=True)
    fixed_config = replace(config, count_inactive_play_bug=False)
    control = run_tensor_feed(bug_config, PERSONALIZED_1PCT)
    treatment = run_tensor_feed(fixed_config, PERSONALIZED_1PCT)
    ab = combine_tensor_ab(control, treatment)
    corrected = treatment["metrics"]["play_rate"] <= 1.0
    reproduced = control["metrics"]["play_rate"] > 1.0
    business_regression = any(
        result["p_value"] < 0.05 and result["relative_lift"] < -0.01
        for name, result in ab.items()
        if name != "negative_rate"
    )
    return {
        "launch_id": "L-BUG-001",
        "category": "bug_fix",
        "title": "Stop counting inactive users as plays",
        "training": "not_applicable_metric_fix",
        "bug_play_rate": control["metrics"]["play_rate"],
        "fixed_play_rate": treatment["metrics"]["play_rate"],
        "ab": ab,
        "shadow_business_metrics_identical": (
            control["metrics"]["stay_per_exposure"]
            == treatment["metrics"]["stay_per_exposure"]
        ),
        "decision": (
            "pass_metric_correction"
            if reproduced and corrected and not business_regression
            else "reject_incomplete_bug_fix"
        ),
    }


def run_system_launches(config: TensorFeedConfig):
    return {
        "config": asdict(config),
        "launches": [
            run_architecture_launch(config),
            run_bug_fix_launch(config),
        ],
    }
