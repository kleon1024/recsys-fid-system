"""Run the POI retrieval launch ladder on a CUDA simulator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ..models.bundle import load_bundle
from .launch import run_retrieval_launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=500_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    bundles = {
        name: load_bundle(args.artifact_dir / f"{name}.pt", args.device)
        for name in ("two_tower", "multi_interest")
    }
    training = json.loads(args.training_report.read_text())
    report = run_retrieval_launch(
        bundles, training, args.users, args.steps, device=args.device
    )
    report["training_report"] = {
        "path": args.training_report.name,
        "sha256": sha256(args.training_report.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([
        {
            "control": row["control"],
            "treatment": row["treatment"],
            "decision": row["decision"],
        }
        for row in report["launches"]
    ], indent=2))


if __name__ == "__main__":
    main()
