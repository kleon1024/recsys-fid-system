"""Build the external KuaiRand sequence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_sequence_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    args = parser.parse_args()
    manifest = build_sequence_dataset(
        args.data_dir, args.output_dir, args.source_commit, args.sequence_length
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "splits": manifest["splits"],
        "sequence_length": manifest["sequence_length"],
    }, indent=2))


if __name__ == "__main__":
    main()
