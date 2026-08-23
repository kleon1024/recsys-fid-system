"""Run multi-queue Load and opportunity-cost iterations on the LT engine."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ...value import DEFAULT_LT_CONFIG
from .local_value_cli import run_repeated_suite, run_suite
from .tensor_engine import DEFAULT_GPU_BATCH_USERS, LOCAL_RETARGET, TensorFeedConfig


QUEUE_POLICIES = (
    replace(
        LOCAL_RETARGET,
        name="organic_local_only",
        multi_queue=True,
        max_ads_per_session=0,
        max_live_per_session=0,
    ),
    replace(
        LOCAL_RETARGET,
        name="add_live_conservative",
        multi_queue=True,
        live_weight=0.05,
        max_ads_per_session=0,
        max_live_per_session=2,
    ),
    replace(
        LOCAL_RETARGET,
        name="add_ad_conservative",
        multi_queue=True,
        live_weight=0.05,
        ad_weight=0.08,
        max_ads_per_session=1,
        min_ad_gap=6,
        max_live_per_session=2,
    ),
    replace(
        LOCAL_RETARGET,
        name="balanced_ad_load",
        multi_queue=True,
        live_weight=0.05,
        ad_weight=0.12,
        max_ads_per_session=2,
        min_ad_gap=4,
        max_live_per_session=2,
    ),
    replace(
        LOCAL_RETARGET,
        name="aggressive_ad_load",
        multi_queue=True,
        live_weight=0.05,
        ad_weight=0.18,
        max_ads_per_session=4,
        min_ad_gap=2,
        max_live_per_session=3,
    ),
)


def apply_exchange_rate_gate(report: dict[str, object]) -> dict[str, object]:
    """Prevent a synthetic commercialization rate from authorizing a launch."""
    rate = DEFAULT_LT_CONFIG.rates["accepted_commercialization_unit"]
    launches = report.get("aggregate", report.get("launches", []))
    for launch in launches:
        metric = launch["metrics"][
            "accepted_platform_commercialization_per_exposure"
        ]
        absolute_lift = (
            metric["pooled_absolute_lift"]
            if "pooled_absolute_lift" in metric
            else metric["treatment_mean"] - metric["control_mean"]
        )
        launch["statistical_decision"] = launch["decision"]
        sensitivity = launch["lt_exchange_sensitivity"]
        zero_rate = sensitivity["0"]
        zero_lift = (
            zero_rate["pooled_absolute_lift"]
            if "pooled_absolute_lift" in zero_rate
            else zero_rate["absolute_lift"]
        )
        zero_p_value = (
            zero_rate["pooled_p_value"]
            if "pooled_p_value" in zero_rate
            else zero_rate["p_value"]
        )
        one_rate = sensitivity["1"]
        one_lift = (
            one_rate["pooled_absolute_lift"]
            if "pooled_absolute_lift" in one_rate
            else one_rate["absolute_lift"]
        )
        observed_commerce_lift = one_lift - zero_lift
        known_zero = (
            zero_rate["known_mean_absolute_effect"]
            if "known_mean_absolute_effect" in zero_rate
            else zero_rate["known_absolute_effect"]
        )
        known_one = (
            one_rate["known_mean_absolute_effect"]
            if "known_mean_absolute_effect" in one_rate
            else one_rate["known_absolute_effect"]
        )
        known_commerce_lift = known_one - known_zero
        launch["exchange_rate_diagnostics"] = {
            "observed_break_even_rate": (
                -zero_lift / observed_commerce_lift
                if abs(observed_commerce_lift) > 1e-12
                else None
            ),
            "known_dgp_break_even_rate": (
                -known_zero / known_commerce_lift
                if abs(known_commerce_lift) > 1e-12
                else None
            ),
            "passes_at_zero_rate": (
                zero_lift > 0.0 and zero_p_value < 0.05
            ),
        }
        if abs(absolute_lift) > 1e-12 and rate.evidence.startswith("synthetic_"):
            launch["decision"] = "hold_exchange_rate_unvalidated"
    report["exchange_rate_review"] = {
        "rate": rate.unit_value,
        "standard_error": rate.standard_error,
        "evidence": rate.evidence,
        "invariant": (
            "A business Value Tree or an unaccepted commercialization rate "
            "cannot authorize an LT launch."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=DEFAULT_GPU_BATCH_USERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = TensorFeedConfig(
        users=arguments.users,
        steps=arguments.steps,
        batch_users=arguments.batch_users,
        device=arguments.device,
    )
    parameters = {
        "policies": QUEUE_POLICIES,
        "suite_name": "multi-queue-load-lt-gpu-v1",
        "launch_prefix": "L-QUEUE-GPU",
    }
    report = apply_exchange_rate_gate(
        run_suite(config, **parameters)
        if arguments.seeds == 1
        else run_repeated_suite(config, arguments.seeds, **parameters)
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
