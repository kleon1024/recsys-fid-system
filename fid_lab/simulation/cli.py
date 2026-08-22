"""Run the stateful recommendation policy-iteration experiment."""

from __future__ import annotations

import argparse
import json

from .contracts import SimulationConfig
from .experiment import run_closed_loop_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2_000)
    parser.add_argument("--items", type=int, default=4_000)
    parser.add_argument("--joiner-users", type=int, default=100)
    parser.add_argument("--include-local", action="store_true")
    args = parser.parse_args()
    report = run_closed_loop_experiment(
        SimulationConfig(users=args.users, items=args.items, joiner_users=args.joiner_users),
        include_local=args.include_local,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
