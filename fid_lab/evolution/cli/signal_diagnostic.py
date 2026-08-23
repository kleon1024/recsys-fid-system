"""Measure how much nonlinear model headroom the declared DGP contains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...scale.contracts import ScaleConfig
from ...scale.synthetic import build_scale_dataset, summarize_distribution
from ..evaluation.model_diagnostics import diagnose_signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impressions", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--signal-version",
        default="industrial-cross-sequence-v1",
        choices=("industrial-cross-sequence-v1", "heterogeneous-nonlinear-v2"),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    dataset = build_scale_dataset(
        ScaleConfig(
            main_impressions=args.impressions,
            seed=args.seed,
            signal_version=args.signal_version,
        )
    )
    report = {
        "config": {
            "main_impressions": args.impressions,
            "seed": args.seed,
            "signal_version": dataset.config.signal_version,
        },
        "distribution": summarize_distribution(dataset),
        "signal_diagnosis": diagnose_signal(dataset, args.device),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
