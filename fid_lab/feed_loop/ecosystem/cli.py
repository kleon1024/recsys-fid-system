"""Run the coupled Feed consumption and creator-supply launch review."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ...feed_posting.contracts import FeedPostingConfig
from ..scale.tensor_engine import TensorFeedConfig
from ..scale.tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from ..scale.tensor_runtime.contracts import EXTERNAL_MIXTURE_FEED_VERSION
from ..tensor_policies import PERSONALIZED, PERSONALIZED_SUPPLY, POPULAR
from .contracts import EcosystemConfig
from .posting import FeedPostingIntervention
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
    parser.add_argument("--creators", type=int, default=25_000)
    parser.add_argument("--catalog-items", type=int, default=200_000)
    parser.add_argument("--posting-batch-creators", type=int, default=25_000)
    parser.add_argument("--max-new-items-per-day", type=int, default=5_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--control", choices=("popular", "personalized"), default="popular"
    )
    parser.add_argument(
        "--treatment", choices=("personalized", "provider_aware"),
        default="personalized",
    )
    parser.add_argument("--supply-weight", type=float, default=0.03)
    parser.add_argument(
        "--experiment-surface", choices=("feed", "feed_posting"),
        default="feed",
    )
    parser.add_argument("--posting-model", type=Path)
    parser.add_argument("--posting-blend", type=float, default=0.20)
    parser.add_argument(
        "--posting-blend-mode", default="legacy_convex"
    )
    args = parser.parse_args()
    total_steps = args.days * args.steps_per_day
    feed = TensorFeedConfig(
        users=args.users, steps=total_steps, candidates=12,
        route_candidates=16, route_oversample=4, merged_candidates=64,
        audit_candidates=32, catalog_items=args.catalog_items,
        catalog_creators=args.creators,
        batch_users=args.batch_users,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        device=args.device, max_sessions=max(args.days * 2, 4),
    )
    objective = (
        "posting_mediation" if args.experiment_surface == "feed_posting"
        else "creator_retention"
        if args.treatment == "provider_aware" else "consumer"
    )
    ecosystem = EcosystemConfig(
        days=args.days, steps_per_day=args.steps_per_day, objective=objective,
        max_new_items_per_day=args.max_new_items_per_day,
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
    control_posting = treatment_posting = None
    if args.experiment_surface == "feed_posting":
        if args.posting_model is None:
            parser.error("--posting-model is required for Feed Posting mediation")
        control = treatment = PERSONALIZED
        posting_config = FeedPostingConfig(
            requests=args.days * feed.catalog_creators,
            creators=feed.catalog_creators,
            prompts=65_536, categories=64, semantic_dim=32,
            sequence_length=64, route_candidates=20,
            merged_candidates=32, exposed_candidates=8,
            world_version="creator-neural-feed-supply-v4",
            device=args.device, catalog_seed=20260824,
        )
        control_posting = FeedPostingIntervention(
            posting_config, args.posting_model, 0.0,
            args.posting_batch_creators, args.posting_blend_mode,
        )
        treatment_posting = FeedPostingIntervention(
            posting_config, args.posting_model, args.posting_blend,
            args.posting_batch_creators, args.posting_blend_mode,
        )
    report = run_ecosystem(
        feed, ecosystem, control, treatment, world,
        control_posting, treatment_posting,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "user_ab": report["user_paired_ab"],
        "creator_ab": report["creator_paired_ab"],
    }, indent=2))


if __name__ == "__main__":
    main()
