"""CLI for architecture and bug-fix launches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...feed_loop.scale.tensor_engine import TensorFeedConfig
from .runner import run_system_launches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_system_launches(
        TensorFeedConfig(users=args.users, steps=args.steps, device=args.device)
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
        print(
            json.dumps(
                [
                    {
                        "launch_id": launch["launch_id"],
                        "decision": launch["decision"],
                    }
                    for launch in report["launches"]
                ],
                indent=2,
            )
        )
    else:
        print(rendered)


if __name__ == "__main__":
    main()
