"""Equal-recall-budget coarse-rank and pass-through launch ladder."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ..scale.local_value_cli import run_repeated_suite, run_suite
from ..scale.tensor_engine import TensorFeedConfig
from ..tensor_policies import LOCAL_INTENT_RANKER


COARSE_POLICIES = (
    replace(
        LOCAL_INTENT_RANKER,
        name="coarse_quality_top20_c0",
        coarse_keep=20,
        coarse_affinity_weight=0.0,
        coarse_quality_weight=1.0,
        coarse_local_weight=0.0,
    ),
    replace(
        LOCAL_INTENT_RANKER,
        name="coarse_lr_top20_c1",
        coarse_keep=20,
        coarse_affinity_weight=1.0,
        coarse_quality_weight=0.35,
        coarse_local_weight=0.0,
    ),
    replace(
        LOCAL_INTENT_RANKER,
        name="coarse_local_cross_top20_c2",
        coarse_keep=20,
        coarse_affinity_weight=1.0,
        coarse_quality_weight=0.35,
        coarse_local_weight=0.15,
    ),
    replace(
        LOCAL_INTENT_RANKER,
        name="coarse_local_cross_top40_c3",
        coarse_keep=40,
        coarse_affinity_weight=1.0,
        coarse_quality_weight=0.35,
        coarse_local_weight=0.15,
    ),
)


def run_cascade_ladder(
    config: TensorFeedConfig,
    seeds: int,
) -> dict[str, object]:
    parameters = {
        "policies": COARSE_POLICIES,
        "suite_name": "main-feed-local-coarse-pass-through-v1",
        "launch_prefix": "L-COARSE-GPU",
    }
    report = (
        run_suite(config, **parameters)
        if seeds == 1
        else run_repeated_suite(config, seeds, **parameters)
    )
    report["stage_contract"] = {
        "recall_candidates": config.candidates,
        "fine_ranker": LOCAL_INTENT_RANKER.name,
        "changed_layer": "coarse_rank_only",
        "metrics": (
            "coarse oracle recall, fine oracle regret, platform LT, Local Value "
            "Tree, and throughput"
        ),
    }
    report["evidence_boundary"] = (
        "These are explicit structural coarse scorers over a synthetic candidate "
        "pool, not trained W&D, DeepFM, or DCNv2 artifacts."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--batch-users", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_cascade_ladder(
        TensorFeedConfig(
            users=arguments.users,
            steps=arguments.steps,
            candidates=arguments.candidates,
            batch_users=arguments.batch_users,
            device=arguments.device,
        ),
        arguments.seeds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
