"""Run adjacent published feature-group LR launches on the GPU tensor engine."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ....simulation.features import FEATURE_GROUP_COLUMNS
from ....value import unified_lt_exchange_report, unified_lt_launch_decision
from ..tensor_engine import TensorFeedConfig, combine_tensor_ab, run_tensor_feed
from .policy import TensorColumnLogisticPolicy


def run_feature_lr_launches(
    report_path: Path,
    artifact_dir: Path,
    config: TensorFeedConfig,
) -> dict[str, object]:
    policies = tuple(
        TensorColumnLogisticPolicy(
            report_path, artifact_dir, group, config.device
        )
        for group in FEATURE_GROUP_COLUMNS
    )
    worlds = {policy.group: run_tensor_feed(config, policy) for policy in policies}
    launches = []
    for control, treatment in zip(policies, policies[1:]):
        metrics = combine_tensor_ab(worlds[control.group], worlds[treatment.group])
        launches.append({
            "launch_id": f"F-LR-{len(launches) + 1:03d}",
            "control": control.group,
            "treatment": treatment.group,
            "added_features": sorted(
                set(treatment.manifest["feature_names"])
                - set(control.manifest["feature_names"])
            ),
            "control_artifact": control.manifest,
            "treatment_artifact": treatment.manifest,
            "ab": metrics,
            "unified_lt_exchange": unified_lt_exchange_report(metrics),
            "decision": unified_lt_launch_decision(metrics["lt_value_per_user"]),
            "diagnostics": {
                "quality_long_view_rate": metrics["quality_long_view_rate"],
                "negative_rate": metrics["negative_rate"],
                "fine_oracle_regret_per_exposure": metrics[
                    "fine_oracle_regret_per_exposure"
                ],
            },
        })
    return {
        "suite": "feature-lr-tensor-launches-v1",
        "config": asdict(config),
        "common_random_numbers": True,
        "launches": launches,
        "world_performance": {
            group: world["performance"] for group, world in worlds.items()
        },
        "evidence_boundary": (
            "Synthetic A/B validates launch mechanics; production readiness "
            "requires accepted causal LT exchange rates and live holdout evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_feature_lr_launches(
        args.report,
        args.artifact_dir,
        TensorFeedConfig(
            users=args.users,
            steps=args.steps,
            batch_users=args.batch_users,
            device=args.device,
            signal_version="heterogeneous-nonlinear-v2",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "launches": [
            {
                "launch_id": launch["launch_id"],
                "change": f"{launch['control']} -> {launch['treatment']}",
                "decision": launch["decision"],
                "lt_relative_lift": launch["ab"]["lt_value_per_user"][
                    "relative_lift"
                ],
            }
            for launch in report["launches"]
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
