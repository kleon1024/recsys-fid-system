"""Train, publish, and fail-closed evaluate the V4 neural world model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from .contracts import WorldModelConfig
from .data import load_world_split
from .training import (
    load_world_ensemble,
    save_world_ensemble,
    train_world_ensemble,
)
from .validation import evaluate_world_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal-evidence", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--reuse-artifact", action="store_true")
    args = parser.parse_args()
    config = replace(
        WorldModelConfig(), epochs=args.epochs, batch_size=args.batch_size,
        ensemble_members=args.ensemble_members,
    )
    test = load_world_split(args.dataset_dir, "test", args.max_eval_rows)
    manifest_path = args.dataset_dir / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text())
    dataset_manifest["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    if args.reuse_artifact:
        ensemble = load_world_ensemble(args.artifact_dir, args.device)
        artifact_manifest = json.loads(
            (args.artifact_dir / "manifest.json").read_text()
        )
        artifact_manifest["manifest_sha256"] = sha256(
            (args.artifact_dir / "manifest.json").read_bytes()
        ).hexdigest()
        training_seconds = 0.0
    else:
        train = load_world_split(args.dataset_dir, "train", args.max_train_rows)
        validation = load_world_split(
            args.dataset_dir, "validation", args.max_eval_rows
        )
        calibration_split = (
            "calibration" if (args.dataset_dir / "calibration.pt").exists()
            else "validation"
        )
        calibration_data = load_world_split(
            args.dataset_dir, calibration_split, args.max_eval_rows
        )
        ensemble, histories, calibration, training_seconds = train_world_ensemble(
            train, validation, config, args.device, calibration_data
        )
        artifact_manifest = save_world_ensemble(
            ensemble, histories, args.artifact_dir, dataset_manifest, calibration
        )
    evaluation = evaluate_world_model(
        ensemble, test, args.device, artifact_manifest["manifest_sha256"],
        args.causal_evidence,
        distribution_rows=min(len(test), 100_000),
        rollout_rows=min(len(test), 10_000),
    )
    report = {
        "schema": "neural-scm-v4-launch-review-v1",
        "dataset_manifest": dataset_manifest,
        "artifact_manifest": artifact_manifest,
        "training_seconds": training_seconds,
        "evaluation": evaluation,
        "authority_transition": (
            "eligible_for_manual_promotion" if evaluation["promotion_eligible"]
            else "v3_remains_executable_authority"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "training_seconds": training_seconds,
        "gates": evaluation["gates"],
        "decision": evaluation["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
