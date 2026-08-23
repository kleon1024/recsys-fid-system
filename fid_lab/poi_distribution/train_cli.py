"""Train the POI distribution model ladder on one request snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from .contracts import PoiDistributionTrainingConfig
from .data import load_exposed_split
from .models.training import save_bundle, train_rankers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = PoiDistributionTrainingConfig(
        epochs=args.epochs, batch_size=args.batch_size, device=args.device
    )
    started = perf_counter()
    train = load_exposed_split(args.dataset_dir, "train")
    validation = load_exposed_split(args.dataset_dir, "validation")
    load_seconds = perf_counter() - started
    models = train_rankers(config, train, validation)
    artifacts = {
        name: save_bundle(
            bundle, args.artifact_dir / f"{name}.pt", config
        )
        for name, bundle in models.items()
    }
    report = {
        "schema": "poi-distribution-model-training-v1",
        "config": asdict(config),
        "dataset_manifest": json.loads(
            (args.dataset_dir / "manifest.json").read_text()
        ),
        "models": {
            name: {**bundle.offline, "artifact": artifacts[name]}
            for name, bundle in models.items()
        },
        "performance": {
            "load_seconds": load_seconds,
            "total_seconds": perf_counter() - started,
        },
        "evidence_boundary": (
            "GPU training on propensity-carrying synthetic V4 request logs; "
            "not external Local business evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "models": list(models), "performance": report["performance"]
    }, indent=2))


if __name__ == "__main__":
    main()
