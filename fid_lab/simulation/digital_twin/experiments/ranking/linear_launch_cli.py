"""CLI for the first learned fine-ranker Launch Review."""

from __future__ import annotations

import argparse
import json

from .linear_launch import LinearRankLaunchConfig, run_linear_rank_launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1_809)
    args = parser.parse_args()
    report = run_linear_rank_launch(LinearRankLaunchConfig(
        dataset_root=args.dataset_root,
        output=args.output,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
    ))
    summary = {
        "offline": report["offline"],
        "decision": report["review"]["decision"],
        "sample": report["review"]["sample"],
    }
    metrics = report["review"]["metrics_per_triggered_user"]
    if "dwell_seconds" in metrics:
        summary["stay"] = metrics["dwell_seconds"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
