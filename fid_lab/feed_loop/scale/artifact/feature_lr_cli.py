"""Run sequential feature proposals against the last accepted LR control."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ....simulation.features import (
    FEATURE_PROPOSAL_COLUMNS,
    feature_set_key,
)
from ....value import unified_lt_exchange_report, unified_lt_launch_decision
from ...release import (
    apply_launch_decision,
    initial_release_state,
    release_state_from_manifest,
    write_release_manifest,
)
from ..tensor_engine import TensorFeedConfig, combine_tensor_ab, run_tensor_feed
from .policy import TensorColumnLogisticPolicy


def run_feature_lr_launches(
    report_path: Path,
    artifact_dir: Path,
    config: TensorFeedConfig,
    prior_release_path: Path | None = None,
) -> dict[str, object]:
    training_report = json.loads(report_path.read_text())
    campaign = training_report.get("campaign")
    if campaign:
        base_key = campaign["base_key"]
        proposals = tuple(campaign["proposals"])
        launch_start = int(campaign["launch_start"])
        report_logical_key = campaign["report_logical_key"]
        artifact_collection = campaign["artifact_collection"]
        suite = "feature-lr-small-sequential-launches-v1"
    else:
        base_key = feature_set_key(())
        proposals = tuple(FEATURE_PROPOSAL_COLUMNS)
        launch_start = 1
        report_logical_key = "feature-lr-sequential-ab"
        artifact_collection = "feature-lr-v2"
        suite = "feature-lr-sequential-launches-v2"
    active_proposals: tuple[str, ...] = ()
    active_key = base_key
    active_policy = TensorColumnLogisticPolicy(
        report_path, artifact_dir, active_key, config.device
    )
    worlds = {active_key: run_tensor_feed(config, active_policy)}
    if prior_release_path:
        prior_release = json.loads(prior_release_path.read_text())
        state = release_state_from_manifest(prior_release, active_key)
        if (
            state["active_artifact"]["artifact_id"]
            != active_policy.manifest["artifact_id"]
        ):
            raise ValueError("campaign base artifact differs from active release")
    else:
        state = initial_release_state(active_key, active_policy.manifest)
    launches = []
    for offset, proposal in enumerate(proposals):
        candidate_proposals = active_proposals + (proposal,)
        candidate_key = base_key + "__" + "__".join(candidate_proposals)
        candidate_policy = TensorColumnLogisticPolicy(
            report_path, artifact_dir, candidate_key, config.device
        )
        worlds[candidate_key] = run_tensor_feed(config, candidate_policy)
        metrics = combine_tensor_ab(worlds[active_key], worlds[candidate_key])
        decision = unified_lt_launch_decision(metrics["lt_value_per_user"])
        launch_id = f"F-LR-{launch_start + offset:03d}"
        state, promotion = apply_launch_decision(
            state,
            candidate_key,
            candidate_policy.manifest,
            decision,
            launch_id,
        )
        launches.append({
            "launch_id": launch_id,
            "proposal": proposal,
            "control": active_key,
            "treatment": candidate_key,
            "added_features": sorted(
                set(candidate_policy.manifest["feature_names"])
                - set(promotion["prior_active_artifact"]["feature_names"])
            ),
            "control_artifact": promotion["prior_active_artifact"],
            "treatment_artifact": candidate_policy.manifest,
            "ab": metrics,
            "unified_lt_exchange": unified_lt_exchange_report(metrics),
            "decision": decision,
            "promotion": promotion,
            "diagnostics": {
                "quality_long_view_rate": metrics["quality_long_view_rate"],
                "negative_rate": metrics["negative_rate"],
                "fine_oracle_regret_per_exposure": metrics[
                    "fine_oracle_regret_per_exposure"
                ],
            },
        })
        if promotion["promoted"]:
            active_proposals = candidate_proposals
            active_key = candidate_key
    return {
        "suite": suite,
        "report_logical_key": report_logical_key,
        "artifact_collection": artifact_collection,
        "config": asdict(config),
        "common_random_numbers": True,
        "control_rule": "every proposal is compared with the last accepted control",
        "launches": launches,
        "release_state": state,
        "production_readiness": launches[-1]["unified_lt_exchange"][
            "production_readiness"
        ],
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
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--prior-release-manifest", type=Path)
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
        args.prior_release_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    release = write_release_manifest(args.output, report, args.release_manifest)
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
        ],
        "active_control": release["active_control_key"],
    }, indent=2))


if __name__ == "__main__":
    main()
