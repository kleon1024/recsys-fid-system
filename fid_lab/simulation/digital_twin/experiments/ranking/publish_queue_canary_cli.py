"""CLI for the expanded Publish Queue canary."""

from __future__ import annotations

import argparse
import json

from .publish_queue_canary import (
    PublishQueueCanaryConfig,
    run_publish_queue_canary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-checkpoint", required=True)
    parser.add_argument("--control-fine-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--burn-in-steps", type=int, default=112)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--ticks-per-day", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1_809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--publish-weight", type=float, default=0.12)
    parser.add_argument("--minimum-triggered-users", type=int, default=30_000)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-wsl-available-gib", type=float, default=4.0)
    parser.add_argument("--minimum-cuda-free-gib", type=float, default=2.0)
    parser.add_argument("--followup-steps", type=int)
    args = parser.parse_args()
    report = run_publish_queue_canary(PublishQueueCanaryConfig(
        publish_checkpoint=args.publish_checkpoint,
        control_fine_checkpoint=args.control_fine_checkpoint,
        output=args.output,
        users=args.users,
        items=args.items,
        burn_in_steps=args.burn_in_steps,
        experiment_steps=args.steps,
        ticks_per_day=args.ticks_per_day,
        seed=args.seed,
        device=args.device,
        publish_weight=args.publish_weight,
        minimum_triggered_users=args.minimum_triggered_users,
        cuda_memory_fraction=args.cuda_memory_fraction,
        minimum_wsl_available_gib=args.minimum_wsl_available_gib,
        minimum_cuda_free_gib=args.minimum_cuda_free_gib,
        followup_steps=args.followup_steps,
    ))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
