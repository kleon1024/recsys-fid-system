"""Run bounded-memory Supply V4 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import PostingWorldConfig
from .runner import run_partitioned_supply_ab, run_partitioned_supply_replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=1_000_000)
    parser.add_argument("--creators", type=int, default=125_000)
    parser.add_argument("--batch-requests", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--catalog-seed", type=int, default=20260824)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--control-model", type=Path)
    parser.add_argument("--partition-dir", type=Path)
    parser.add_argument("--ab", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PostingWorldConfig(
        requests=args.requests,
        creators=args.creators,
        batch_requests=args.batch_requests,
        seed=args.seed,
        catalog_seed=args.catalog_seed,
        world_version="creator-neural-supply-v4",
        device=args.device,
    )
    if args.ab:
        if args.model is None:
            raise ValueError("partitioned A/B requires --model")
        report = run_partitioned_supply_ab(
            config, args.model, args.batch_requests, args.control_model
        )
    else:
        report = run_partitioned_supply_replay(
            config, args.model, args.batch_requests, args.partition_dir
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "schema": report["schema"],
        "policy": report.get("policy", report.get("treatment")),
        "partitions": len(report.get("partitions", ())),
        "resume": report.get("resume"),
        "performance": report["performance"],
    }, indent=2))


if __name__ == "__main__":
    main()
