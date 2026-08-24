"""Run one fixed ranker artifact through held-out hidden worlds."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..experimentation.robustness import run_heldout_environment_gate
from ..profiles import PROFILE_OVERRIDES, load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OVERRIDES), default="screen"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--architecture", default="mmoe")
    parser.add_argument("--source-environment-seed", type=int, default=20260825)
    parser.add_argument(
        "--heldout-environment-seeds", default="20261834,20262843"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = replace(
        load_profile(args.profile, args.device),
        environment_seed=args.source_environment_seed,
    )
    heldout = tuple(
        int(value) for value in args.heldout_environment_seeds.split(",")
    )
    report = run_heldout_environment_gate(
        config,
        architecture=args.architecture,
        heldout_environment_seeds=heldout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "artifact_fingerprint": report["artifact_fingerprint"],
        "aggregate_decision": report["aggregate_decision"],
        "world_decisions": [
            row["decision"] for row in report["evaluations"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
