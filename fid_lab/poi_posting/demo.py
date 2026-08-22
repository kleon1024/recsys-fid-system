"""Run the complete local POI posting model experiment."""

from __future__ import annotations

import json

from .training import run_experiment


def main() -> None:
    print(json.dumps(run_experiment().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
