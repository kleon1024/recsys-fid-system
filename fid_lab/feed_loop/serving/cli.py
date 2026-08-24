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
from .contracts import CompositeValueTreeConfig
from .launch import run_composite_serving_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-artifact", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--behavior-dataset", type=Path, required=True)
    parser.add_argument("--feed-report", type=Path, required=True)
    parser.add_argument("--feed-artifact-dir", type=Path, required=True)
    parser.add_argument("--local-artifact", type=Path, required=True)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-coarse-weight", type=float, default=0.025)
    parser.add_argument("--local-fine-weight", type=float, default=0.025)
    parser.add_argument("--local-coarse-keep", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users, steps=args.steps, batch_users=args.batch_users,
        candidates=12, route_candidates=16, route_oversample=4,
        merged_candidates=64, audit_candidates=32,
        catalog_items=200_000, catalog_creators=25_000,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        device=args.device, retain_paired_user_metrics=True,
    )
    world = ExternalSequenceMixtureWorld(
        args.behavior_artifact, args.calibration_report,
        args.behavior_dataset, args.device, config.seed,
    )
    feed = TensorV4RequestPolicy(
        "mmoe", args.feed_report, args.feed_artifact_dir,
        args.device, blend_weight=0.01, base_tolerance=0.05,
    )
    local = load_bundle(args.local_artifact, args.device)
    value_config = CompositeValueTreeConfig(
        local_coarse_weight=args.local_coarse_weight,
        local_fine_weight=args.local_fine_weight,
        local_coarse_keep=args.local_coarse_keep,
    )
    report = run_composite_serving_launch(
        config, feed, local, world, value_config
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
