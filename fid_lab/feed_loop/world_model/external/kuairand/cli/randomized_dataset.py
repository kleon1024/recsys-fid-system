"""Build the checksum-bound KuaiRand-1K randomized evidence dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.randomized import build_randomized_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-record", required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    manifest = build_randomized_dataset(
        args.data_dir, args.output_dir, args.source_record,
        args.sequence_length, args.seed,
    )
    print(json.dumps({
        "schema": manifest["schema"],
        "splits": manifest["splits"],
        "random_logging_propensity": manifest["random_logging_propensity"],
    }, indent=2))


if __name__ == "__main__":
    main()
