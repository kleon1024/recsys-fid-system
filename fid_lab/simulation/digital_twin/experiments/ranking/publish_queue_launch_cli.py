"""CLI for the independent Feed Publish Queue launch review."""

from __future__ import annotations

import argparse
import json

from .publish_queue_launch import (
    PublishQueueLaunchConfig,
    run_publish_queue_launch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--control-fine-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--burn-in-steps", type=int, default=112)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--ticks-per-day", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1_809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--sparse-hash-size", type=int, default=1 << 18)
    parser.add_argument("--publish-weight", type=float, default=0.12)
    args = parser.parse_args()
    report = run_publish_queue_launch(PublishQueueLaunchConfig(
        dataset_root=args.dataset_root,
        control_fine_checkpoint=args.control_fine_checkpoint,
        output=args.output,
        users=args.users,
        items=args.items,
        burn_in_steps=args.burn_in_steps,
        experiment_steps=args.steps,
        ticks_per_day=args.ticks_per_day,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        sparse_hash_size=args.sparse_hash_size,
        publish_weight=args.publish_weight,
    ))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
