"""Materialize the external request-level NeuralSCM bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.core_bridge import build_core_bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()
    manifest = build_core_bridge(
        args.dataset_dir, args.output_dir, args.candidates
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "splits": manifest["splits"],
        "evidence_boundary": manifest["evidence_boundary"],
    }, indent=2))


if __name__ == "__main__":
    main()
