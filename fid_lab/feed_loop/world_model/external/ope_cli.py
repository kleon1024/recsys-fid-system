"""Run randomized full-corpus doubly robust policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ope import run_randomized_ope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--control-artifact", type=Path, required=True)
    parser.add_argument("--treatment-artifact", type=Path, required=True)
    parser.add_argument("--world-artifact", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=8_192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--uniform-mixture", type=float, default=0.50)
    parser.add_argument("--treatment-calibration-report", type=Path)
    parser.add_argument(
        "--treatment-calibration-key", default="sequence_transformer"
    )
    parser.add_argument(
        "--utility-mode", choices=("raw_probability", "standardized_feed"),
        default="raw_probability",
    )
    parser.add_argument("--minimum-standard-exposures", type=int, default=5)
    args = parser.parse_args()
    report = run_randomized_ope(
        args.dataset_dir, args.control_artifact, args.treatment_artifact,
        args.world_artifact, args.benchmark_report, args.device, args.rows,
        args.batch_size, args.seed, args.temperature, args.uniform_mixture,
        args.treatment_calibration_report,
        args.treatment_calibration_key,
        args.utility_mode,
        args.minimum_standard_exposures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "metrics": report["metrics"],
        "importance_sampling": report["importance_sampling"],
    }, indent=2))


if __name__ == "__main__":
    main()
