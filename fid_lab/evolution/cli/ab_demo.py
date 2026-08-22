"""Print simulated online business lift for product, model, and strategy changes."""

from __future__ import annotations

import argparse
import json

from ..evaluation.ab_simulator import run_scenario_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=200_000)
    args = parser.parse_args()
    print(json.dumps(run_scenario_suite(args.users), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
