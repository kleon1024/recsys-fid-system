"""Run the continuous events-to-training-to-A/B digital-twin loop."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..profiles import PROFILE_OVERRIDES, load_profile
from ..training.orchestrator import (
    ContinuousLearningConfig,
    run_continuous_learning,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OVERRIDES), default="screen"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ladder = ("lr", "wide_deep", "dcnv2", "mmoe")
    architectures = tuple(
        ladder[min(index, len(ladder) - 1)]
        for index in range(args.iterations)
    )
    blend_weights = tuple(
        min(0.10 * (index + 1), 0.50) for index in range(args.iterations)
    )
    twin_config = load_profile(args.profile, args.device)
    if args.environment_seed is not None:
        twin_config = replace(
            twin_config, environment_seed=args.environment_seed
        )
    report = run_continuous_learning(
        twin_config,
        ContinuousLearningConfig(
            iterations=args.iterations,
            architectures=architectures,
            fine_model_weights=blend_weights,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "final_world_step": report["final_world_step"],
        "active_stack": report["active_stack"]["name"],
        "decisions": [row["decision"] for row in report["iterations"]],
        "registry": report["registry"],
    }, indent=2))


if __name__ == "__main__":
    main()
