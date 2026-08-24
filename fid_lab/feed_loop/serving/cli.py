"""Run the unified Feed, Local Value Tree, and mixer review on GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...poi_distribution.models.training import load_bundle
from ..scale.model_ladder.v4.serving import TensorV4RequestPolicy
from ..scale.tensor_engine import TensorFeedConfig
from ..scale.tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from ..scale.tensor_runtime.contracts import EXTERNAL_MIXTURE_FEED_VERSION
from ..scale.tensor_runtime.contracts import LOCAL_NEURAL_SIGNAL_VERSION
from .contracts import CompositeValueTreeConfig
from ..governance.contracts import ContentGovernanceConfig
from .launch import run_composite_serving_launch


DEFAULT_GOVERNANCE = ContentGovernanceConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-artifact", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--behavior-dataset", type=Path, required=True)
    parser.add_argument("--feed-report", type=Path, required=True)
    parser.add_argument("--feed-artifact-dir", type=Path, required=True)
    parser.add_argument("--local-artifact", type=Path, required=True)
    parser.add_argument("--control-local-artifact", type=Path)
    parser.add_argument("--coarse-local-artifact", type=Path)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feed-inference-chunk", type=int, default=2_048)
    parser.add_argument("--experiment-salt", type=int, default=0x1B873593)
    parser.add_argument("--local-coarse-weight", type=float, default=0.025)
    parser.add_argument("--local-fine-weight", type=float, default=0.025)
    parser.add_argument("--local-coarse-keep", type=int, default=20)
    parser.add_argument("--content-governance", action="store_true")
    parser.add_argument(
        "--governance-max-risk", type=float,
        default=DEFAULT_GOVERNANCE.max_predicted_integrity_risk,
    )
    parser.add_argument(
        "--governance-cluster-penalty", type=float,
        default=DEFAULT_GOVERNANCE.repeated_cluster_penalty,
    )
    parser.add_argument(
        "--governance-author-penalty", type=float,
        default=DEFAULT_GOVERNANCE.repeated_author_penalty,
    )
    parser.add_argument(
        "--governance-creator-boost", type=float,
        default=DEFAULT_GOVERNANCE.new_creator_boost,
    )
    parser.add_argument(
        "--governance-max-poi", type=int,
        default=DEFAULT_GOVERNANCE.max_poi_per_session,
    )
    parser.add_argument(
        "--governance-min-poi-gap", type=int,
        default=DEFAULT_GOVERNANCE.min_poi_gap,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users, steps=args.steps + args.warmup_steps,
        batch_users=args.batch_users,
        candidates=12, route_candidates=16, route_oversample=4,
        merged_candidates=64, audit_candidates=32,
        catalog_items=200_000, catalog_creators=25_000,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        local_signal_version=LOCAL_NEURAL_SIGNAL_VERSION,
        device=args.device, retain_paired_user_metrics=True,
        experiment_salt=args.experiment_salt,
    )
    world = ExternalSequenceMixtureWorld(
        args.behavior_artifact, args.calibration_report,
        args.behavior_dataset, args.device, config.seed,
    )
    feed = TensorV4RequestPolicy(
        "mmoe", args.feed_report, args.feed_artifact_dir,
        args.device, blend_weight=0.01, base_tolerance=0.05,
        inference_chunk=args.feed_inference_chunk,
    )
    local = load_bundle(args.local_artifact, args.device)
    control_local = (
        None if args.control_local_artifact is None
        else load_bundle(args.control_local_artifact, args.device)
    )
    coarse_local = (
        None if args.coarse_local_artifact is None
        else load_bundle(args.coarse_local_artifact, args.device)
    )
    value_config = CompositeValueTreeConfig(
        local_coarse_weight=args.local_coarse_weight,
        local_fine_weight=args.local_fine_weight,
        local_coarse_keep=args.local_coarse_keep,
    )
    report = run_composite_serving_launch(
        config, feed, local, world, value_config,
        control_local_bundle=control_local,
        treatment_coarse_local_bundle=coarse_local,
        warmup_steps=args.warmup_steps,
        treatment_governance_config=(
            ContentGovernanceConfig(
                max_predicted_integrity_risk=args.governance_max_risk,
                repeated_cluster_penalty=args.governance_cluster_penalty,
                repeated_author_penalty=args.governance_author_penalty,
                new_creator_boost=args.governance_creator_boost,
                max_poi_per_session=args.governance_max_poi,
                min_poi_gap=args.governance_min_poi_gap,
            ) if args.content_governance else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "shadow_gates": report["shadow_gates"],
        "online_gates": report["online_gates"],
        "performance": report["performance"],
    }, indent=2))


if __name__ == "__main__":
    main()
