"""Run the trained POI distribution model ladder on the RTX 4090."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import MODEL_NAMES
from .launch import run_poi_distribution_launch
from .models.training import load_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stages", nargs="+",
        choices=("coarse", "fine", "mix", "end_to_end"),
        default=("coarse", "fine", "mix"),
    )
    args = parser.parse_args()
    bundles = {
        name: load_bundle(args.artifact_dir / f"{name}.pt", args.device)
        for name in MODEL_NAMES
    }
    report = run_poi_distribution_launch(
        bundles, users=args.users, steps=args.steps, device=args.device,
        stages=tuple(args.stages),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([
        {
            "stage": row["stage"], "control": row["control"],
            "treatment": row["treatment"], "decision": row["decision"],
        }
        for row in report["launches"]
    ], indent=2))


if __name__ == "__main__":
    main()
