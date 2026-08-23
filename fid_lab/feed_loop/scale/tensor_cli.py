"""CLI for GPU-resident main Feed simulation throughput."""

from __future__ import annotations

import argparse
import json

from .tensor_engine import (
    DEFAULT_GPU_BATCH_USERS,
    PERSONALIZED,
    PERSONALIZED_1PCT,
    POPULAR,
    TensorFeedConfig,
    combine_tensor_ab,
    run_tensor_feed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=DEFAULT_GPU_BATCH_USERS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users,
        steps=args.steps,
        batch_users=args.batch_users,
        device=args.device,
    )
    reports = {
        policy.name: run_tensor_feed(config, policy)
        for policy in (POPULAR, PERSONALIZED_1PCT, PERSONALIZED)
    }
    reports["ab_1pct_trigger_vs_baseline"] = combine_tensor_ab(
        reports[POPULAR.name], reports[PERSONALIZED_1PCT.name]
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
