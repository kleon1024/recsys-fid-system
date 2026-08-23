"""CLI for the V3 propensity-carrying request log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import V3LoggingConfig, build_v3_logging_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    manifest = build_v3_logging_dataset(
        root,
        args.output_dir,
        V3LoggingConfig(
            users=args.users, steps=args.steps, batch_users=args.batch_users,
            epsilon=args.epsilon, device=args.device,
        ),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
