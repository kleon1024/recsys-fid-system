"""CLI for the creator-clustered Posting ranker Launch Review."""

from __future__ import annotations

import argparse
import json

from .posting_launch import PostingRankLaunchConfig, run_posting_rank_launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-fine-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1_809)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    report = run_posting_rank_launch(PostingRankLaunchConfig(
        candidate_fine_checkpoint=args.candidate_fine_checkpoint,
        output=args.output,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        experiment_steps=args.steps,
    ))
    print(json.dumps(report["review"], indent=2))


if __name__ == "__main__":
    main()
