"""Post-ramp reverse holdout for long-horizon Local model value."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from math import erfc, sqrt
from pathlib import Path

from ..scale.local_value_cli import (
    decision_from_metrics,
    enrich_power_diagnostics,
    pool_exchange_sensitivity,
    pooled_decision,
)
from ..scale.lt_exchange import combine_lt_exchange_sensitivity
from ..scale.tensor_engine import (
    DEFAULT_GPU_BATCH_USERS,
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)
from ..tensor_policies import LOCAL_INTENT_RANKER, LOCAL_RETARGET


@dataclass(frozen=True)
class ReverseHoldoutConfig:
    users: int = 1_000_000
    steps: int = 48
    burn_in_steps: int = 12
    seeds: int = 3
    batch_users: int = DEFAULT_GPU_BATCH_USERS
    seed: int = 20260823
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not 0 < self.burn_in_steps < self.steps:
            raise ValueError("burn-in must leave a non-empty holdout window")
        if self.seeds < 1:
            raise ValueError("reverse holdout requires at least one seed")


def _known_effect(control, treatment, metrics):
    output = {}
    for name in metrics:
        zero = control["metrics"][name]
        one = treatment["metrics"][name]
        output[name] = {
            "absolute_effect": one - zero,
            "relative_effect": (one - zero) / zero if abs(zero) > 1e-12 else None,
        }
    return output


def _run_seed(config: ReverseHoldoutConfig, seed: int):
    tensor_config = TensorFeedConfig(
        users=config.users,
        steps=config.steps,
        batch_users=config.batch_users,
        seed=seed,
        device=config.device,
    )
    reverted = (
        (LOCAL_INTENT_RANKER,) * config.burn_in_steps
        + (LOCAL_RETARGET,) * (config.steps - config.burn_in_steps)
    )
    continuous = (LOCAL_INTENT_RANKER,) * config.steps
    control = run_tensor_feed(
        tensor_config,
        LOCAL_INTENT_RANKER,
        policy_schedule=reverted,
        measurement_start_step=config.burn_in_steps,
    )
    treatment = run_tensor_feed(
        tensor_config,
        LOCAL_INTENT_RANKER,
        policy_schedule=continuous,
        measurement_start_step=config.burn_in_steps,
    )
    metrics = combine_tensor_ab(control, treatment)
    launch = {
        "launch_id": "L-LOCAL-REVERSE-001",
        "control": "revert_to_local_search_retarget_v3",
        "treatment": "continue_local_intent_quality_rank_v4",
        "metrics": metrics,
        "lt_exchange_sensitivity": combine_lt_exchange_sensitivity(
            control, treatment
        ),
        "known_dgp_effect": _known_effect(control, treatment, metrics),
        "decision": decision_from_metrics(metrics),
        "control_performance": control["performance"],
        "treatment_performance": treatment["performance"],
    }
    return {"config": asdict(tensor_config), "launches": [launch]}


def _aggregate_metrics(replicates):
    summary = {}
    launch_index = 0
    metric_names = replicates[0]["launches"][launch_index]["metrics"]
    for name in metric_names:
        cells = [
            replicate["launches"][launch_index]["metrics"][name]
            for replicate in replicates
        ]
        weights = [1.0 / max(cell["standard_error"] ** 2, 1e-24) for cell in cells]
        weight_total = sum(weights)
        effect = sum(
            weight * (cell["treatment_mean"] - cell["control_mean"])
            for weight, cell in zip(weights, cells)
        ) / weight_total
        control = sum(
            weight * cell["control_mean"] for weight, cell in zip(weights, cells)
        ) / weight_total
        standard_error = sqrt(1.0 / weight_total)
        observed = [cell["relative_lift"] for cell in cells]
        known = [
            replicate["launches"][launch_index]["known_dgp_effect"][name][
                "relative_effect"
            ]
            for replicate in replicates
        ]
        observed = [value for value in observed if value is not None]
        known = [value for value in known if value is not None]
        summary[name] = {
            "observed_mean_relative_lift": (
                sum(observed) / len(observed) if observed else None
            ),
            "known_mean_relative_effect": sum(known) / len(known) if known else None,
            "positive_observed_seeds": sum(value > 0.0 for value in observed),
            "positive_truth_seeds": sum(value > 0.0 for value in known),
            "pooled_absolute_lift": effect,
            "pooled_relative_lift": effect / control if abs(control) > 1e-12 else None,
            "pooled_standard_error": standard_error,
            "pooled_p_value": erfc(
                abs(effect / max(standard_error, 1e-12)) / sqrt(2.0)
            ),
        }
    return summary


def _reverse_decision(decision: str) -> str:
    if decision == "pass_lt_value":
        return "retain_launch_long_horizon"
    if decision.startswith("reject_"):
        return "rollback_launch"
    return "maintain_reverse_holdout"


def run_reverse_holdout(
    config: ReverseHoldoutConfig = ReverseHoldoutConfig(),
) -> dict[str, object]:
    replicates = [
        _run_seed(config, config.seed + offset) for offset in range(config.seeds)
    ]
    metrics = _aggregate_metrics(replicates)
    aggregate = {
        "launch_id": "L-LOCAL-REVERSE-001",
        "control": "revert_to_local_search_retarget_v3",
        "treatment": "continue_local_intent_quality_rank_v4",
        "metrics": metrics,
        "lt_exchange_sensitivity": pool_exchange_sensitivity(replicates, 0),
        "seed_decisions": [
            replicate["launches"][0]["decision"] for replicate in replicates
        ],
    }
    aggregate["statistical_decision"] = pooled_decision(metrics)
    aggregate["decision"] = _reverse_decision(aggregate["statistical_decision"])
    report = {
        "suite": "local-intent-ranker-reverse-holdout-v1",
        "protocol": {
            "burn_in_steps": config.burn_in_steps,
            "measurement_steps": config.steps - config.burn_in_steps,
            "control": "new model during burn-in, then reverted old model",
            "treatment": "new model during burn-in and measurement",
        },
        "config": asdict(config),
        "aggregate": [aggregate],
        "replicates": replicates,
        "evidence_boundary": (
            "Synthetic reverse holdout validates protocol and power; it is not "
            "a company production lift estimate."
        ),
    }
    return enrich_power_diagnostics(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--burn-in-steps", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-users", type=int, default=DEFAULT_GPU_BATCH_USERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_reverse_holdout(
        ReverseHoldoutConfig(
            users=arguments.users,
            steps=arguments.steps,
            burn_in_steps=arguments.burn_in_steps,
            seeds=arguments.seeds,
            batch_users=arguments.batch_users,
            device=arguments.device,
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
