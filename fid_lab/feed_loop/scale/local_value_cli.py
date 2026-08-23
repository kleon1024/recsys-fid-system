"""Run the Local Service LT launch ladder on the tensor engine."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from math import ceil, erfc, sqrt
from pathlib import Path

from ...value import unified_lt_launch_decision
from .lt_exchange import combine_lt_exchange_sensitivity
from .tensor_engine import (
    DEFAULT_GPU_BATCH_USERS,
    LOCAL_EXPANSION,
    LOCAL_INTENT_RANKER,
    LOCAL_RETARGET,
    LOCAL_SEARCH,
    LOCAL_STATIC,
    PERSONALIZED,
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)


def decision_from_metrics(metrics: dict[str, dict[str, float]]) -> str:
    return unified_lt_launch_decision(metrics["lt_value_per_user"])


def run_suite(
    config: TensorFeedConfig,
    policies=None,
    suite_name: str = "local-service-main-feed-lt-gpu-v1",
    launch_prefix: str = "L-LOCAL-GPU",
) -> dict[str, object]:
    policies = policies or (
        PERSONALIZED,
        LOCAL_STATIC,
        LOCAL_SEARCH,
        LOCAL_RETARGET,
        LOCAL_INTENT_RANKER,
        LOCAL_EXPANSION,
    )
    worlds = {policy.name: run_tensor_feed(config, policy) for policy in policies}
    launches = []
    for index, (control, treatment) in enumerate(zip(policies, policies[1:]), start=1):
        metrics = combine_tensor_ab(worlds[control.name], worlds[treatment.name])
        known_effect = {}
        for name in metrics:
            control_mean = worlds[control.name]["metrics"][name]
            treatment_mean = worlds[treatment.name]["metrics"][name]
            known_effect[name] = {
                "absolute_effect": treatment_mean - control_mean,
                "relative_effect": (
                    (treatment_mean - control_mean) / control_mean
                    if abs(control_mean) > 1e-12
                    else None
                ),
            }
        launches.append(
            {
                "launch_id": f"{launch_prefix}-{index:03d}",
                "control": control.name,
                "treatment": treatment.name,
                "metrics": metrics,
                "lt_exchange_sensitivity": combine_lt_exchange_sensitivity(
                    worlds[control.name], worlds[treatment.name]
                ),
                "known_dgp_effect": known_effect,
                "decision": decision_from_metrics(metrics),
                "control_performance": worlds[control.name]["performance"],
                "treatment_performance": worlds[treatment.name]["performance"],
            }
        )
    return {
        "suite": suite_name,
        "config": asdict(config),
        "launches": launches,
        "lt_contract": "lt-platform-metrics-v1",
        "evidence_boundary": "Synthetic DGP effects are not production lift estimates.",
    }


def run_repeated_suite(
    config: TensorFeedConfig,
    seeds: int,
    policies=None,
    suite_name: str = "local-service-main-feed-lt-gpu-repeated-v1",
    launch_prefix: str = "L-LOCAL-GPU",
) -> dict[str, object]:
    replicates = [
        run_suite(
            replace(config, seed=config.seed + index),
            policies,
            suite_name,
            launch_prefix,
        )
        for index in range(seeds)
    ]
    aggregate = []
    for launch_index, launch in enumerate(replicates[0]["launches"]):
        metric_summary = {}
        for metric in launch["metrics"]:
            cells = [
                replicate["launches"][launch_index]["metrics"][metric]
                for replicate in replicates
            ]
            observed = [
                replicate["launches"][launch_index]["metrics"][metric]["relative_lift"]
                for replicate in replicates
            ]
            truth = [
                replicate["launches"][launch_index]["known_dgp_effect"][metric][
                    "relative_effect"
                ]
                for replicate in replicates
            ]
            observed_values = [value for value in observed if value is not None]
            truth_values = [value for value in truth if value is not None]
            inverse_variance = [1.0 / max(cell["standard_error"] ** 2, 1e-24) for cell in cells]
            pooled_effect = sum(
                weight * (cell["treatment_mean"] - cell["control_mean"])
                for weight, cell in zip(inverse_variance, cells)
            ) / sum(inverse_variance)
            pooled_standard_error = sqrt(1.0 / sum(inverse_variance))
            pooled_control = sum(
                weight * cell["control_mean"]
                for weight, cell in zip(inverse_variance, cells)
            ) / sum(inverse_variance)
            metric_summary[metric] = {
                "observed_mean_relative_lift": (
                    sum(observed_values) / len(observed_values) if observed_values else None
                ),
                "known_mean_relative_effect": (
                    sum(truth_values) / len(truth_values) if truth_values else None
                ),
                "positive_observed_seeds": sum(value > 0.0 for value in observed_values),
                "positive_truth_seeds": sum(value > 0.0 for value in truth_values),
                "pooled_absolute_lift": pooled_effect,
                "pooled_relative_lift": (
                    pooled_effect / pooled_control if abs(pooled_control) > 1e-12 else None
                ),
                "pooled_standard_error": pooled_standard_error,
                "pooled_p_value": erfc(
                    abs(pooled_effect / max(pooled_standard_error, 1e-12)) / sqrt(2.0)
                ),
            }
        decisions = [
            replicate["launches"][launch_index]["decision"] for replicate in replicates
        ]
        aggregate_launch = {
                "launch_id": launch["launch_id"],
                "control": launch["control"],
                "treatment": launch["treatment"],
                "decisions": decisions,
                "metrics": metric_summary,
                "lt_exchange_sensitivity": pool_exchange_sensitivity(
                    replicates, launch_index
                ),
            }
        aggregate_launch["decision"] = pooled_decision(metric_summary)
        aggregate.append(aggregate_launch)
    return enrich_power_diagnostics({
        "suite": suite_name,
        "seeds": seeds,
        "base_config": asdict(config),
        "aggregate": aggregate,
        "replicates": replicates,
    })


def enrich_power_diagnostics(report):
    """Add MDE and required-user evidence without rerunning trajectories."""
    if "aggregate" not in report:
        return report
    current_total_users = sum(
        replicate["config"]["users"] for replicate in report["replicates"]
    )
    for launch_index, launch in enumerate(report["aggregate"]):
        for name, metric in launch["metrics"].items():
            truths = [
                replicate["launches"][launch_index]["known_dgp_effect"][name][
                    "absolute_effect"
                ]
                for replicate in report["replicates"]
            ]
            truth = sum(truths) / len(truths)
            standard_error = metric["pooled_standard_error"]
            metric["known_mean_absolute_effect"] = truth
            metric["mde_absolute_80pct_power"] = 2.80 * standard_error
            metric["required_total_users_for_known_effect_80pct_power"] = (
                None
                if abs(truth) < 1e-12
                else ceil(
                    current_total_users
                    * (2.80 * standard_error / abs(truth)) ** 2
                )
            )
    return report


def pool_exchange_sensitivity(replicates, launch_index):
    output = {}
    scenario_names = replicates[0]["launches"][launch_index][
        "lt_exchange_sensitivity"
    ]
    for name in scenario_names:
        cells = [
            replicate["launches"][launch_index]["lt_exchange_sensitivity"][name]
            for replicate in replicates
        ]
        weights = [1.0 / max(cell["standard_error"] ** 2, 1e-24) for cell in cells]
        weight_total = sum(weights)
        effect = sum(
            weight * cell["absolute_lift"] for weight, cell in zip(weights, cells)
        ) / weight_total
        control = sum(
            weight * cell["control_mean"] for weight, cell in zip(weights, cells)
        ) / weight_total
        standard_error = sqrt(1.0 / weight_total)
        known = [cell["known_relative_effect"] for cell in cells]
        known_absolute = [cell["known_absolute_effect"] for cell in cells]
        output[name] = {
            "exchange_rate": cells[0]["exchange_rate"],
            "pooled_absolute_lift": effect,
            "pooled_relative_lift": effect / control if abs(control) > 1e-12 else None,
            "pooled_standard_error": standard_error,
            "pooled_confidence_interval": (
                effect - 1.96 * standard_error,
                effect + 1.96 * standard_error,
            ),
            "pooled_p_value": erfc(
                abs(effect / max(standard_error, 1e-12)) / sqrt(2.0)
            ),
            "known_mean_relative_effect": sum(known) / len(known),
            "known_mean_absolute_effect": sum(known_absolute)
            / len(known_absolute),
        }
    return output


def pooled_decision(metrics: dict[str, dict[str, float]]) -> str:
    return unified_lt_launch_decision(metrics["lt_value_per_user"], pooled=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=DEFAULT_GPU_BATCH_USERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--intent-only", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = TensorFeedConfig(
            users=arguments.users,
            steps=arguments.steps,
            batch_users=arguments.batch_users,
            device=arguments.device,
        )
    parameters = (
        {
            "policies": (
                LOCAL_RETARGET,
                LOCAL_INTENT_RANKER,
                LOCAL_EXPANSION,
            ),
            "suite_name": "local-intent-ranker-scale-gpu-v1",
            "launch_prefix": "L-LOCAL-SCALE",
        }
        if arguments.intent_only
        else {}
    )
    report = (
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
