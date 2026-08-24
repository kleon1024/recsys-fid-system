"""Run the GPU Feed-posting launch ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import FeedPostingConfig
from .launch import run_repeated_feed_posting_ladder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=150_000)
    parser.add_argument("--prompts", type=int, default=32_768)
    parser.add_argument("--categories", type=int, default=32)
    parser.add_argument("--semantic-dim", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--route-candidates", type=int, default=12)
    parser.add_argument("--merged-candidates", type=int, default=20)
    parser.add_argument("--exposed-candidates", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--generation-batch-requests", type=int, default=50_000)
    parser.add_argument("--creators", type=int, default=25_000)
    parser.add_argument("--catalog-seed", type=int)
    parser.add_argument(
        "--world-version", default="teacher-hidden-feed-posting-v1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260824, 20260825, 20260826)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    config = FeedPostingConfig(
        requests=args.requests, train_epochs=args.epochs,
        prompts=args.prompts, categories=args.categories,
        semantic_dim=args.semantic_dim, sequence_length=args.sequence_length,
        route_candidates=args.route_candidates,
        merged_candidates=args.merged_candidates,
        exposed_candidates=args.exposed_candidates,
        generation_batch_requests=args.generation_batch_requests,
        creators=args.creators, catalog_seed=args.catalog_seed,
        world_version=args.world_version,
        device=args.device, seed=args.seeds[0],
    )
    report = run_repeated_feed_posting_ladder(
        config, tuple(args.seeds), args.artifact_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "release_state": report["release_state"],
        "decisions": [row["decision"] for row in report["launches"]],
    }, indent=2))


if __name__ == "__main__":
    main()
