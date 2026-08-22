"""CLI for independent main-Feed launch candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...feed_loop.scale.tensor_engine import TensorFeedConfig
from .runner import run_policy_launch_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=25_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_policy_launch_suite(
        TensorFeedConfig(
            users=args.users,
            steps=args.steps,
            batch_users=args.batch_users,
            device=args.device,
        )
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
        print(
            json.dumps(
                [
                    {
                        "launch_id": value["spec"]["launch_id"],
                        "decision": value["decision"],
                        "primary": value["spec"]["primary_metric"],
                        "lift": value["ab"][value["spec"]["primary_metric"]][
                            "relative_lift"
                        ],
                        "p_value": value["ab"][value["spec"]["primary_metric"]][
                            "p_value"
                        ],
                    }
                    for value in result["launches"]
                ],
                indent=2,
            )
        )
    else:
        print(rendered)


if __name__ == "__main__":
    main()
