"""CLI for the V4 request-aware Feed model ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_request_ladder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-rows", type=int, default=200_000)
    parser.add_argument("--validation-rows", type=int, default=60_000)
    parser.add_argument("--test-rows", type=int, default=60_000)
    parser.add_argument("--ranking-rows", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    report = run_request_ladder(
        args.dataset_dir, args.artifact_dir, args.device,
        args.train_rows, args.validation_rows, args.test_rows,
        args.ranking_rows, args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        name: {
            "auc": value["offline"]["auc"],
            "gauc": value["offline"]["user_gauc"]["value"],
            "regret": value["ranking"]["audit_regret"],
            "seconds": value["training_seconds"],
        }
        for name, value in report["models"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
