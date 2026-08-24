"""Run the coupled Feed consumption and creator-supply launch review."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ..scale.tensor_engine import TensorFeedConfig
from ..scale.tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from ..scale.tensor_runtime.contracts import EXTERNAL_MIXTURE_FEED_VERSION
from ..tensor_policies import PERSONALIZED, PERSONALIZED_SUPPLY, POPULAR
from .contracts import EcosystemConfig
from .runner import run_ecosystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--steps-per-day", type=int, default=8)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--control", choices=("popular", "personalized"), default="popular"
    )
    parser.add_argument(
        "--treatment", choices=("personalized", "provider_aware"),
        default="personalized",
    )
    parser.add_argument("--supply-weight", type=float, default=0.03)
    args = parser.parse_args()
    total_steps = args.days * args.steps_per_day
    feed = TensorFeedConfig(
        users=args.users, steps=total_steps, candidates=12,
        route_candidates=16, route_oversample=4, merged_candidates=64,
        audit_candidates=32, catalog_items=200_000, catalog_creators=25_000,
        batch_users=args.batch_users,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        device=args.device, max_sessions=max(args.days * 2, 4),
    )
    objective = (
        "creator_retention"
        if args.treatment == "provider_aware" else "consumer"
    )
    ecosystem = EcosystemConfig(
        days=args.days, steps_per_day=args.steps_per_day, objective=objective,
    )
    world = ExternalSequenceMixtureWorld(
        args.artifact, args.calibration_report, args.dataset_dir,
        args.device, feed.seed,
    )
    control = POPULAR if args.control == "popular" else PERSONALIZED
    treatment = (
        PERSONALIZED if args.treatment == "personalized"
        else replace(
            PERSONALIZED_SUPPLY,
            creator_supply_weight=args.supply_weight,
            name=f"personalized_provider_aware_{args.supply_weight:g}",
        )
    )
    report = run_ecosystem(feed, ecosystem, control, treatment, world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "user_ab": report["user_paired_ab"],
        "creator_ab": report["creator_paired_ab"],
    }, indent=2))


if __name__ == "__main__":
    main()
