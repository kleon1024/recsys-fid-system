"""Run the multi-surface digital twin and persist its review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..contracts import TwinPolicy
from ..experimentation.experiment import run_twin_experiment
from ..profiles import PROFILE_OVERRIDES, load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OVERRIDES), default="gpu"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-salt", type=int, default=0x1B873593)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_profile(args.profile, args.device)
    treatment = TwinPolicy(
        name="history_aware_v2",
        realtime_weight=0.30,
        author_fatigue_penalty=0.06,
        cluster_fatigue_penalty=0.09,
        topic_fatigue_penalty=0.05,
    )
    experiment = run_twin_experiment(
        config, treatment_policy=treatment,
        experiment_salt=args.experiment_salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(experiment.report, indent=2) + "\n")
    print(json.dumps({
        "schema": experiment.report["schema"],
        "synthetic_lt": experiment.report["cuped_ab"][
            "synthetic_lt_measurement"
        ],
        "trace_gates": experiment.report["trace"]["gates"],
        "performance": experiment.report["performance"],
    }, indent=2))


if __name__ == "__main__":
    main()
