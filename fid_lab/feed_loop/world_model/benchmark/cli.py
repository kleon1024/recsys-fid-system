"""CLI for the V4 equal-request model-capacity benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_model_capacity_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-rows", type=int, default=300_000)
    parser.add_argument("--eval-rows", type=int, default=100_000)
    parser.add_argument("--request-rows", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    report = run_model_capacity_benchmark(
        args.dataset_dir, args.artifact_dir, args.device,
        args.train_rows, args.eval_rows, args.request_rows, args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "gates": report["gates"],
        "models": {
        name: {
            "auc": metrics["auc"],
            "oracle_regret": metrics["request"]["oracle_regret"],
        }
        for name, metrics in report["models"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
