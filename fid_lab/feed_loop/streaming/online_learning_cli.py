"""CLI for the Joiner-to-PS main-Feed online learning launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .online_learning import run_online_learning_launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000)
    parser.add_argument("--items", type=int, default=4_000)
    parser.add_argument("--ab-users", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_online_learning_launch(
        args.users, args.items, args.ab_users, args.epochs
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
        launches = report["launches"]
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "ps_version": report["parameter_server"]["version"],
                    "shadow_delta": report["shadow_replay_score_delta"],
                    "launches": {
                        name: {
                            "decision": value["decision"],
                            "stay_lift": value["ab"]["stay_per_exposure"][
                                "relative_lift"
                            ],
                            "stay_p": value["ab"]["stay_per_exposure"]["p_value"],
                        }
                        for name, value in launches.items()
                    },
                },
                indent=2,
            )
        )
    else:
        print(rendered)


if __name__ == "__main__":
    main()
