"""Run oracle, feature-ablation, and model-coupled A/B diagnostics."""

from __future__ import annotations

import argparse
import json

from ...scale.contracts import ScaleConfig
from ...scale.synthetic import build_scale_dataset
from ..evaluation.model_ab import run_model_coupled_ab
from ..evaluation.model_diagnostics import diagnose_signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-impressions", type=int, default=1_000_000)
    args = parser.parse_args()
    dataset = build_scale_dataset(ScaleConfig(main_impressions=args.main_impressions))
    print(
        json.dumps(
            {
                "signal_diagnostics": diagnose_signal(dataset),
                "model_coupled_ab": run_model_coupled_ab(dataset),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
