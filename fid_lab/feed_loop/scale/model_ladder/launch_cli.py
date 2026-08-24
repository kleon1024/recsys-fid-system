"""Sequential V3 shadow, A/B, LT gate, and release-state review."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ....value import unified_lt_launch_decision
from ...tensor_policies import PERSONALIZED
from ..tensor_engine import TensorFeedConfig, combine_tensor_ab, run_tensor_feed
from .serving import TensorV3ModelPolicy


MODEL_ORDER = (
    "lr_v3_long_view", "xgboost_v3_long_view", "wide_deep", "deepfm",
    "dcnv2", "mmoe_value_tree",
    "mmoe_feed_multitask_stay_v2",
)

GUARDED_VARIANTS = (
    ("mmoe_multitask_guarded_a005_t003", 0.05, 0.03),
    ("mmoe_multitask_guarded_a010_t005", 0.10, 0.05),
    ("mmoe_multitask_guarded_a015_t008", 0.15, 0.08),
)


def run_ladder(training_report, artifact_dir, config, model_order=MODEL_ORDER,
               include_guarded=True):
    policies = {
        name: TensorV3ModelPolicy(
            name, training_report, artifact_dir, config.device
        )
        for name in model_order
    }
    guarded = GUARDED_VARIANTS if include_guarded else ()
    for deployment_name, blend_weight, tolerance in guarded:
        policies[deployment_name] = TensorV3ModelPolicy(
            "mmoe_feed_multitask_stay_v2", training_report, artifact_dir, config.device,
            deployment_name, blend_weight, tolerance,
        )
    worlds = {"rule_personalized_v1": run_tensor_feed(config, PERSONALIZED)}
    active_key = "rule_personalized_v1"
    launches = []
    order = (*model_order, *(name for name, _, _ in guarded))
    for index, name in enumerate(order, start=1):
        candidate = policies[name]
        worlds[name] = run_tensor_feed(config, candidate)
        metrics = combine_tensor_ab(worlds[active_key], worlds[name])
        decision = unified_lt_launch_decision(metrics["lt_value_per_user"])
        duration_shift = metrics["selected_duration_per_exposure"]["relative_lift"]
        if (
            decision == "pass_unified_lt_nonnegative"
            and duration_shift is not None
            and abs(duration_shift) > 0.05
        ):
            decision = "hold_reward_hacking_duration_shift"
        promoted = decision == "pass_unified_lt_nonnegative"
        launches.append({
            "launch_id": f"V3-MODEL-{index:03d}",
            "control": active_key,
            "treatment": name,
            "artifact_manifest": candidate.manifest,
            "serving_policy": candidate.describe(),
            "shadow_replay_max_delta": json.loads(training_report.read_text())[
                "models"
            ][candidate.model_name]["shadow_replay_max_delta"],
            "ab": metrics,
            "decision": decision,
            "promoted": promoted,
            "diagnostics": {
                "stay": metrics["stay_per_exposure"],
                "long_view": metrics["long_view_rate"],
                "quality_long_view": metrics["quality_long_view_rate"],
                "negative": metrics["negative_rate"],
                "oracle_regret": metrics["fine_oracle_regret_per_exposure"],
                "selected_duration": metrics["selected_duration_per_exposure"],
            },
            "performance": worlds[name]["performance"],
        })
        if promoted:
            active_key = name
    active_manifest = (
        None if active_key == "rule_personalized_v1" else policies[active_key].manifest
    )
    return {
        "schema": "v3-sequential-model-launches-v1",
        "config": asdict(config),
        "control_rule": "each candidate compares with the last accepted control",
        "launches": launches,
        "release_state": {
            "active_key": active_key,
            "active_artifact": active_manifest,
            "rollback_key": "rule_personalized_v1" if active_manifest else None,
        },
        "world_performance": {
            name: world["performance"] for name, world in worlds.items()
        },
        "evidence_boundary": "Synthetic V3 A/B, not production lift evidence.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=50_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--multitask-only", action="store_true")
    parser.add_argument("--no-guarded", action="store_true")
    args = parser.parse_args()
    report = run_ladder(
        args.training_report,
        args.artifact_dir,
        TensorFeedConfig(
            users=args.users, steps=args.steps, batch_users=args.batch_users,
            device=args.device, signal_version="kuairand-calibrated-v3",
        ),
        (("mmoe_feed_multitask_stay_v2",) if args.multitask_only else MODEL_ORDER),
        not args.no_guarded,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "launches": [
            {
                "id": launch["launch_id"], "control": launch["control"],
                "treatment": launch["treatment"], "decision": launch["decision"],
                "lt": launch["ab"]["lt_value_per_user"]["relative_lift"],
            }
            for launch in report["launches"]
        ],
        "active": report["release_state"]["active_key"],
    }, indent=2))


if __name__ == "__main__":
    main()
