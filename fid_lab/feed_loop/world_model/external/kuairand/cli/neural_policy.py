"""CLI for content-bound NeuralSCM randomized Feed policy evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.neural_policy import run_neural_policy_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=8_192)
    parser.add_argument("--action-samples", type=int, default=64)
    parser.add_argument("--request-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    report = run_neural_policy_evidence(
        args.dataset_dir, args.bridge_dir, args.artifact_dir,
        args.device, args.rows, args.action_samples,
        args.request_batch, args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "gates": report["gates"],
        "policy_kendall_tau": report["policy_kendall_tau"],
        "identified_policy_pairs": report["identified_policy_pairs"],
    }, indent=2))


if __name__ == "__main__":
    main()
