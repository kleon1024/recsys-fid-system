"""Run sequential feature proposals against the last accepted LR control."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from ..tensor_engine import (
    DEFAULT_GPU_BATCH_USERS,
    TensorFeedConfig,
    combine_tensor_ab,
    combine_tensor_trigger_ab,
    run_tensor_feed,
)
from .policy import TensorColumnLogisticPolicy


def _run_world(config, policy, burn_policy, burn_in, trigger_kind):
    schedule = None
    if burn_in:
        schedule = (burn_policy,) * burn_in + (policy,) * (config.steps - burn_in)
    return run_tensor_feed(
        config,
        policy,
        policy_schedule=schedule,
        measurement_start_step=burn_in,
        trigger_kind=trigger_kind,
    )


def _evaluate_proposal(
    config,
    worlds,
    active_key,
    active_policy,
    candidate_key,
    candidate_policy,
    trigger_kind,
    burn_in,
):
    if trigger_kind:
        control = _run_world(
            config, active_policy, active_policy, burn_in, trigger_kind
        )
        treatment = _run_world(
            config, candidate_policy, active_policy, burn_in, trigger_kind
        )
        worlds[f"{active_key}@{trigger_kind}"] = control
        worlds[f"{candidate_key}@{trigger_kind}"] = treatment
        trigger_analysis = combine_tensor_trigger_ab(control, treatment)
    else:
        if active_key not in worlds:
            worlds[active_key] = run_tensor_feed(config, active_policy)
        worlds[candidate_key] = run_tensor_feed(config, candidate_policy)
        control = worlds[active_key]
        treatment = worlds[candidate_key]
        trigger_analysis = None
    return combine_tensor_ab(control, treatment), trigger_analysis


def _campaign_settings(training_report, config):
    campaign = training_report.get("campaign")
    if campaign:
        updated = replace(
            config, search_event_rate=float(campaign.get("search_event_rate", 0.0))
        )
        return (
            campaign["base_key"], tuple(campaign["proposals"]),
            int(campaign["launch_start"]), campaign["report_logical_key"],
            campaign["artifact_collection"], campaign["operation"],
            campaign.get("trigger_kinds", {}),
            int(campaign.get("burn_in_steps", 0)), updated,
            "feature-lr-small-sequential-launches-v1",
        )
    return (
        feature_set_key(()), tuple(FEATURE_PROPOSAL_COLUMNS), 1,
        "feature-lr-sequential-ab", "feature-lr-v2", "add", {}, 0, config,
        "feature-lr-sequential-launches-v2",
    )


def run_feature_lr_launches(
    report_path: Path,
    artifact_dir: Path,
    config: TensorFeedConfig,
    prior_release_path: Path | None = None,
) -> dict[str, object]:
    training_report = json.loads(report_path.read_text())
    settings = _campaign_settings(training_report, config)
    (
        base_key, proposals, launch_start, report_logical_key,
        artifact_collection, operation, trigger_kinds, burn_in, config, suite,
    ) = settings
    active_proposals: tuple[str, ...] = ()
    active_key = base_key
    active_policy = TensorColumnLogisticPolicy(
        report_path, artifact_dir, active_key, config.device
    )
    worlds = {}
    if not trigger_kinds:
        worlds[active_key] = run_tensor_feed(config, active_policy)
    if prior_release_path:
        prior_release = json.loads(prior_release_path.read_text())
        state = release_state_from_manifest(prior_release, active_key)
        if (
            state["active_artifact"]["artifact_id"]
            != active_policy.manifest["artifact_id"]
        ):
            raise ValueError("campaign base artifact differs from active release")
    else:
        state = initial_release_state(
            active_key, active_policy.manifest, artifact_collection
        )
    launches = []
    for offset, proposal in enumerate(proposals):
        candidate_proposals = active_proposals + (proposal,)
        candidate_key = base_key + "__" + "__".join(candidate_proposals)
        candidate_policy = TensorColumnLogisticPolicy(
            report_path, artifact_dir, candidate_key, config.device
        )
        trigger_kind = trigger_kinds.get(proposal)
        metrics, trigger_analysis = _evaluate_proposal(
            config,
            worlds,
            active_key,
            active_policy,
            candidate_key,
            candidate_policy,
            trigger_kind,
            burn_in,
        )
        decision = unified_lt_launch_decision(metrics["lt_value_per_user"])
        launch_id = f"F-LR-{launch_start + offset:03d}"
        state, promotion = apply_launch_decision(
            state,
            candidate_key,
            candidate_policy.manifest,
            decision,
            launch_id,
            artifact_collection,
        )
        prior_features = set(promotion["prior_active_artifact"]["feature_names"])
        candidate_features = set(candidate_policy.manifest["feature_names"])
        launches.append({
            "launch_id": launch_id,
            "proposal": proposal,
            "control": active_key,
            "treatment": candidate_key,
            "added_features": sorted(
                candidate_features - prior_features
            ),
            "removed_features": sorted(prior_features - candidate_features),
            "control_artifact": promotion["prior_active_artifact"],
            "treatment_artifact": candidate_policy.manifest,
            "ab": metrics,
            "trigger_analysis": trigger_analysis,
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
            active_policy = candidate_policy
    return {
        "suite": suite,
        "report_logical_key": report_logical_key,
        "artifact_collection": artifact_collection,
        "operation": operation,
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
    parser.add_argument("--batch-users", type=int, default=DEFAULT_GPU_BATCH_USERS)
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
