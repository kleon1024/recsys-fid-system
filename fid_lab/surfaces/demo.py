"""Train all recommendation-surface reference models."""

from __future__ import annotations

import json

from .experiment import run_surface_suite


def main() -> None:
    print(json.dumps(run_surface_suite(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
