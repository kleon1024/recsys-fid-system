"""Run one fast factual retrieval Launch Review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .retrieval_ladder import RetrievalLadderConfig, run_retrieval_ladder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_retrieval_ladder(RetrievalLadderConfig(
        users=args.users, items=args.items, burn_in_steps=8,
        experiment_steps=args.steps, control_fraction=0.45,
        treatment_fraction=0.45, minimum_triggered_users=500,
        max_reviews=1, max_attempts_per_review=1,
        response_authority_mode="formula_oracle", device=args.device,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    review = report["reviews"][0]
    print(json.dumps({
        "decision": review["decision"],
        "reason": review["reason"],
        "sample": review["sample"],
        "dwell": review["metrics_per_triggered_user"]["dwell_seconds"],
        "negative": review["metrics_per_triggered_user"]["negative"],
    }, indent=2))


if __name__ == "__main__":
    main()
