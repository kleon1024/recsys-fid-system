"""Run a powered Feed Posting creator A/B on GPU partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import FeedPostingConfig
from .runner import run_partitioned_feed_posting_ab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=10_000_000)
    parser.add_argument("--creators", type=int, default=1_250_000)
    parser.add_argument("--partition-requests", type=int, default=50_000)
    parser.add_argument("--prompts", type=int, default=200_000)
    parser.add_argument("--categories", type=int, default=64)
    parser.add_argument("--semantic-dim", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--route-candidates", type=int, default=40)
    parser.add_argument("--merged-candidates", type=int, default=80)
    parser.add_argument("--exposed-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--catalog-seed", type=int, default=20260824)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--control-model", type=Path)
    parser.add_argument("--treatment-blend", type=float, default=1.0)
    parser.add_argument("--control-blend", type=float, default=1.0)
    parser.add_argument(
        "--treatment-blend-mode", default="legacy_convex"
    )
    parser.add_argument("--control-blend-mode", default="legacy_convex")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_partitioned_feed_posting_ab(
        FeedPostingConfig(
            requests=args.requests, creators=args.creators,
            prompts=args.prompts, categories=args.categories,
            semantic_dim=args.semantic_dim, sequence_length=args.sequence_length,
            route_candidates=args.route_candidates,
            merged_candidates=args.merged_candidates,
            exposed_candidates=args.exposed_candidates,
            seed=args.seed, catalog_seed=args.catalog_seed,
            world_version="creator-neural-feed-supply-v4", device=args.device,
        ),
        args.model, args.partition_requests, args.control_model,
        args.treatment_blend, args.control_blend,
        args.treatment_blend_mode, args.control_blend_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "control": report["control"],
        "treatment": report["treatment"],
        "decision": report["decision"],
        "performance": report["performance"],
    }, indent=2))


if __name__ == "__main__":
    main()
