"""Materialize the external-mixture V4 request-level candidate dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from ..tensor_runtime.contracts import EXTERNAL_MIXTURE_FEED_VERSION
from .dataset import V3LoggingConfig, build_v4_logging_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--external-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-users", type=int, default=2_000)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    config = V3LoggingConfig(
        users=args.users, steps=args.steps, batch_users=args.batch_users,
        epsilon=args.epsilon, sequence_length=64, device=args.device,
        signal_version=EXTERNAL_MIXTURE_FEED_VERSION,
        candidates=12, route_candidates=16, route_oversample=4,
        merged_candidates=64, audit_candidates=32,
        catalog_items=200_000, catalog_creators=25_000,
    )
    world = ExternalSequenceMixtureWorld(
        args.artifact, args.calibration_report, args.external_dataset_dir,
        args.device, config.seed, inference_batch=min(args.batch_users, 512),
    )
    manifest = build_v4_logging_dataset(root, args.output_dir, config, world)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
