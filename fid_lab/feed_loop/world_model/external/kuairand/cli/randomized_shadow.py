"""CLI adapter for catalog-aware stateful randomized shadow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.shadow import run_randomized_shadow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--control-artifact", type=Path, required=True)
    parser.add_argument("--treatment-artifact", type=Path, required=True)
    parser.add_argument("--world-artifact", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--world-adapter-report", type=Path)
    parser.add_argument("--world-calibration-key", default="independent_world")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--minimum-standard-exposures", type=int, default=5)
    parser.add_argument("--simulated-ab-users", type=int, default=100_000)
    args = parser.parse_args()
    report = run_randomized_shadow(
        args.dataset_dir, args.control_artifact, args.treatment_artifact,
        args.world_artifact, args.benchmark_report, args.adapter_report,
        args.world_adapter_report,
        args.world_calibration_key,
        device=args.device, rows=args.rows, candidates=args.candidates,
        steps=args.steps, batch_size=args.batch_size, seed=args.seed,
        minimum_standard_exposures=args.minimum_standard_exposures,
        simulated_ab_users=args.simulated_ab_users,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "metrics": report["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
