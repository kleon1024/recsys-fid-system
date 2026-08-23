"""Run the GPU POI distribution stage ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage_ladder import run_poi_stage_ladder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260824, 20260825, 20260826))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_poi_stage_ladder(
        args.users, args.steps, tuple(args.seeds), args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "stage_decisions": report["stage_decisions"],
        "launches": len(report["launches"]),
    }, indent=2))


if __name__ == "__main__":
    main()
