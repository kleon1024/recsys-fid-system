"""Run the RTX 4090 POI Detail Launch Review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import PoiDetailConfig
from .launch import run_repeated_poi_detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-requests", type=int, default=4_096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260824, 20260825, 20260826)
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PoiDetailConfig(
        requests=args.requests, users=max(args.requests // 2, 1),
        train_epochs=args.epochs, train_batch_requests=args.batch_requests,
        device=args.device, seed=args.seeds[0],
    )
    report = run_repeated_poi_detail(
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
