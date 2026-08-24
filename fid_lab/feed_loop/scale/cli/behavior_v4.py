"""Run and publish the external-mixture Feed V4 behavior review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..tensor_engine import TensorFeedConfig
from ..tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from ..tensor_runtime.behavior.review import run_feed_behavior_review
from ..tensor_runtime.contracts import (
    EXTERNAL_MIXTURE_FEED_VERSION,
    LOCAL_NEURAL_SIGNAL_VERSION,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--catalog-items", type=int, default=200_000)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users, steps=args.steps, candidates=12,
        route_candidates=16, route_oversample=4, merged_candidates=64,
        audit_candidates=32, catalog_items=args.catalog_items,
        batch_users=args.batch_users,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        local_signal_version=LOCAL_NEURAL_SIGNAL_VERSION,
        device=args.device, retain_paired_user_metrics=True,
    )
    world = ExternalSequenceMixtureWorld(
        args.artifact, args.calibration_report, args.dataset_dir,
        args.device, config.seed,
    )
    report = run_feed_behavior_review(config, world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "behavior_world": report["behavior_world"],
    }, indent=2))


if __name__ == "__main__":
    main()
