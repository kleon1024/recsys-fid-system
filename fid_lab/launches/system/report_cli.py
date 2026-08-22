"""Render system Launch Reviews from immutable evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from .reporting import render_system_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in render_system_suite(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
