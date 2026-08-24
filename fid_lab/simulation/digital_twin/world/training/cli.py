"""CLI for held-out v4 structural NeuralSCM data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bridge import build_structural_bridge
from .contracts import StructuralBridgeConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300_000)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=200_000)
    parser.add_argument("--ticks", type=int, default=192)
    parser.add_argument("--test-family-id", type=int, default=5)
    parser.add_argument("--reuse-build", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = build_structural_bridge(
        args.output_dir,
        StructuralBridgeConfig(
            rows=args.rows,
            users=args.users,
            items=args.items,
            ticks=args.ticks,
            test_family_id=args.test_family_id,
            device=args.device,
        ),
        args.reuse_build,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
