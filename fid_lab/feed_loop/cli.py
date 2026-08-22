"""Train and A/B the main-Feed model ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_feed_model_ladder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000)
    parser.add_argument("--items", type=int, default=4_000)
    parser.add_argument("--ab-users", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_feed_model_ladder(
        args.users, args.items, args.ab_users, args.epochs, args.device
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
        summary = {
            "examples": report["examples"],
            "positive_rate": report["positive_rate"],
            "offline": {
                name: {
                    "auc": result["auc"],
                    "gauc": result["user_gauc"]["value"],
                    "oracle_regret": result["candidate"]["oracle_regret"],
                }
                for name, result in report["offline"].items()
            },
            "launches": {
                name: {
                    "decision": result["decision"],
                    "true_stay_per_exposure_lift": result["metrics"]["stay_per_exposure"][
                        "true_relative_itt"
                    ],
                    "observed_stay_per_exposure_p": result["metrics"][
                        "stay_per_exposure"
                    ]["p_value"],
                }
                for name, result in report["launches"].items()
            },
        }
        print(json.dumps(summary, indent=2))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
