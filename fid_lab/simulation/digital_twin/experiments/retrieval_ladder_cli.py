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
    parser.add_argument("--burn-in-steps", type=int, default=112)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--initial-route", default="random")
    parser.add_argument("--candidate-route", default="popular")
    parser.add_argument("--aa-steps", type=int, default=8)
    args = parser.parse_args()
    report = run_retrieval_ladder(RetrievalLadderConfig(
        users=args.users, items=args.items, burn_in_steps=args.burn_in_steps,
        experiment_steps=args.steps, control_fraction=0.45,
        treatment_fraction=0.45, minimum_triggered_users=500,
        max_reviews=1, max_attempts_per_review=1,
        response_authority_mode="formula_oracle", ticks_per_day=16,
        launch_bundle_root=str(
            args.bundle_root or args.output.with_suffix("")
        ),
        seed=args.seed,
        device=args.device,
        initial_route=args.initial_route,
        route_ladder=(args.candidate_route,),
        aa_steps=args.aa_steps,
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
