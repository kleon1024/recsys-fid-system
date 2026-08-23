"""CLI for the public KuaiRand calibration profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kuairand import build_kuairand_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_kuairand_calibration(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["profile"], indent=2))


if __name__ == "__main__":
    main()
