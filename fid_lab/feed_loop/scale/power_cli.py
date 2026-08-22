"""CLI for the vectorized per-mille A/B power check."""

from __future__ import annotations

import argparse
import json

from .small_effect_ab import run_small_effect_ab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    args = parser.parse_args()
    print(json.dumps(run_small_effect_ab(args.users), indent=2))


if __name__ == "__main__":
    main()
