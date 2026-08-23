"""Train and publish the same-snapshot V3 model ladder."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import exposed_split, load_tensors
from .evaluate import (
    candidate_metrics,
    multitask_metrics,
    offline_metrics,
    rule_candidate_metrics,
    rule_offline_metrics,
)
from .train import publish_models, train_models


def _artifact_manifest(artifact_dir):
    lines = []
    for path in sorted(artifact_dir.iterdir()):
        if path.name == "MANIFEST.sha256" or not path.is_file():
            continue
        lines.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}")
    manifest = artifact_dir / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    return sha256(manifest.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train = exposed_split(args.dataset_dir, "train")
    validation = exposed_split(args.dataset_dir, "validation")
    test = exposed_split(args.dataset_dir, "test")
    models, timing = train_models(
        train, validation, args.epochs, args.device, 20260823
    )
    published, replay = publish_models(
        models, validation.features[:20_000], args.artifact_dir,
        "kuairand-calibrated-v3",
    )
    test_tensors = load_tensors(args.dataset_dir, "test")
    results = {
        "rule_personalized_v1": {
            "offline": rule_offline_metrics(test),
            "candidate": rule_candidate_metrics(test_tensors),
            "training_seconds": 0.0,
            "artifact_manifest": None,
            "shadow_replay_max_delta": 0.0,
        }
    }
    for model in models:
        results[model.name] = {
            "offline": offline_metrics(model, test),
            "candidate": candidate_metrics(model, test_tensors),
            "training_seconds": timing[model.name],
            "artifact_manifest": dict(published[model.name].artifact_manifest),
            "shadow_replay_max_delta": replay[model.name],
            "parameters": getattr(model, "parameters", None),
            "loss_history": getattr(model, "loss_history", None),
        }
        if model.name.startswith("mmoe_feed_multitask"):
            results[model.name]["primitive_heads"] = multitask_metrics(model, test)
    report = {
        "schema": "v3-model-ladder-training-v1",
        "dataset_manifest": json.loads(
            (args.dataset_dir / "manifest.json").read_text()
        ),
        "target": "single-target baselines plus primitive multi-task challenger",
        "propensity_correction": "exact_ips_or_exact_weighted_resampling",
        "epochs": args.epochs,
        "models": results,
        "artifact_manifest_sha256": _artifact_manifest(args.artifact_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        name: {
            "auc": value["offline"]["auc"],
            "gauc": value["offline"]["user_gauc"]["value"],
            "regret": value["candidate"]["audit_regret"],
            "seconds": value["training_seconds"],
        }
        for name, value in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
