"""Train, publish, and fail-closed evaluate the V4 neural world model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from .contracts import WorldModelConfig
from .data import concatenate_world_splits, load_world_split
from .training import (
    adapt_world_ensemble,
    calibrate_world_ensemble,
    load_world_ensemble,
    save_world_ensemble,
    train_world_ensemble,
)
from .validation import evaluate_world_model
from .validation.support import fit_support_profile


def _load_dataset_manifest(dataset_dir):
    path = dataset_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["manifest_sha256"] = sha256(path.read_bytes()).hexdigest()
    return manifest


def _combined_dataset_manifest(primary, auxiliary):
    if auxiliary is None:
        return primary
    if primary.get("feature_contract_sha256") != auxiliary.get(
        "feature_contract_sha256"
    ):
        raise ValueError("primary and auxiliary feature contracts differ")
    coverage = {}
    keys = set(primary.get("feature_coverage", ())) | set(
        auxiliary.get("feature_coverage", ())
    )
    for key in sorted(keys, key=int):
        left = primary.get("feature_coverage", {}).get(key, "unavailable")
        right = auxiliary.get("feature_coverage", {}).get(key, "unavailable")
        coverage[key] = (
            "multi_source" if right == "native_v4" and "observed" in left
            else "native_v4" if right == "native_v4" else left
        )
    manifest = {
        "schema": "multi-source-neural-scm-dataset-v1",
        "source_manifest_sha256s": [
            primary["manifest_sha256"], auxiliary["manifest_sha256"],
        ],
        "feature_contract_sha256": primary["feature_contract_sha256"],
        "feature_coverage": coverage,
        "evidence_boundary": (
            "External rows own Feed distribution and randomized policy evidence; "
            "synthetic rows own structural stress coverage only."
        ),
    }
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest["manifest_sha256"] = sha256(encoded).hexdigest()
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-dataset-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-artifact-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-evidence", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--reuse-artifact", action="store_true")
    parser.add_argument("--adapt-artifact", action="store_true")
    parser.add_argument("--recalibrate-artifact", action="store_true")
    parser.add_argument("--refit-support-profile", action="store_true")
    args = parser.parse_args()
    if args.recalibrate_artifact and not args.reuse_artifact:
        raise ValueError("artifact recalibration requires --reuse-artifact")
    if args.adapt_artifact and not args.reuse_artifact:
        raise ValueError("artifact adaptation requires --reuse-artifact")
    if args.source_artifact_dir is not None and not args.reuse_artifact:
        raise ValueError("source artifact requires --reuse-artifact")
    if args.refit_support_profile and args.source_artifact_dir is None:
        raise ValueError("support refit requires --source-artifact-dir")
    if args.refit_support_profile and (
        args.source_artifact_dir.resolve() == args.artifact_dir.resolve()
    ):
        raise ValueError("support refit target must differ from source artifact")
    return args


def _assert_artifact_dataset(artifact, dataset):
    expected_sources = dataset.get(
        "source_manifest_sha256s", [dataset.get("manifest_sha256")],
    )
    if artifact.get("dataset_source_manifest_sha256s") != expected_sources:
        raise ValueError("world artifact dataset sources do not match evaluation")
    if artifact.get("feature_contract_sha256") != dataset.get(
        "feature_contract_sha256"
    ):
        raise ValueError("world artifact feature contract does not match evaluation")
    if artifact.get("feature_coverage") != dataset.get("feature_coverage"):
        raise ValueError("world artifact feature coverage does not match evaluation")


def _reuse_artifact(args, dataset_manifest):
    source_dir = args.source_artifact_dir or args.artifact_dir
    if source_dir != args.artifact_dir and args.artifact_dir.exists():
        raise ValueError("target artifact directory already exists")
    manifest_path = source_dir / "manifest.json"
    artifact_manifest = json.loads(manifest_path.read_text())
    _assert_artifact_dataset(artifact_manifest, dataset_manifest)
    ensemble = load_world_ensemble(source_dir, args.device)
    artifact_manifest["manifest_sha256"] = sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    adaptation = None
    if args.adapt_artifact:
        if not (args.dataset_dir / "adaptation.pt").exists():
            raise ValueError("artifact adaptation requires adaptation.pt")
        adaptation = adapt_world_ensemble(
            ensemble,
            load_world_split(args.dataset_dir, "adaptation", args.max_eval_rows),
            next(ensemble.parameters()).device,
        )
    if (
        args.recalibrate_artifact or args.adapt_artifact
        or args.refit_support_profile
    ):
        calibration_split = (
            "calibration" if (args.dataset_dir / "calibration.pt").exists()
            else "validation"
        )
        calibration = artifact_manifest.get("calibration")
        if args.recalibrate_artifact or args.adapt_artifact:
            calibration = calibrate_world_ensemble(
                ensemble,
                load_world_split(
                    args.dataset_dir, calibration_split, args.max_eval_rows,
                ),
                next(ensemble.parameters()).device,
            )
        if adaptation is not None:
            calibration["randomized_adaptation"] = adaptation
        support_profile = artifact_manifest.get("support_profile")
        if args.refit_support_profile:
            _, _, train_sources, validation_sources = _training_splits(args)
            support_profile = fit_support_profile(
                train_sources, validation_sources,
            )
        artifact_manifest = save_world_ensemble(
            ensemble, artifact_manifest.get("training_history", []),
            args.artifact_dir, dataset_manifest, calibration,
            support_profile,
        )
    return ensemble, artifact_manifest, 0.0


def _training_splits(args):
    train_sources = [
        load_world_split(args.dataset_dir, "train", args.max_train_rows),
    ]
    validation_sources = [load_world_split(
        args.dataset_dir, "validation", args.max_eval_rows,
    )]
    if args.auxiliary_dataset_dir is not None:
        train_sources.append(load_world_split(
            args.auxiliary_dataset_dir, "train", args.max_train_rows,
        ))
        validation_sources.append(load_world_split(
            args.auxiliary_dataset_dir, "validation", args.max_eval_rows,
        ))
    return (
        concatenate_world_splits(tuple(train_sources)),
        concatenate_world_splits(tuple(validation_sources)),
        tuple(train_sources), tuple(validation_sources),
    )


def _train_artifact(args, config, dataset_manifest):
    train, validation, train_sources, validation_sources = _training_splits(args)
    calibration_split = (
        "calibration" if (args.dataset_dir / "calibration.pt").exists()
        else "validation"
    )
    calibration_data = load_world_split(
        args.dataset_dir, calibration_split, args.max_eval_rows,
    )
    adaptation_data = (
        load_world_split(args.dataset_dir, "adaptation", args.max_eval_rows)
        if (args.dataset_dir / "adaptation.pt").exists() else None
    )
    structural_path = (
        None if args.auxiliary_dataset_dir is None
        else args.auxiliary_dataset_dir / "structural_adaptation.pt"
    )
    structural_data = (
        load_world_split(
            args.auxiliary_dataset_dir, "structural_adaptation",
            args.max_train_rows,
        ) if structural_path is not None and structural_path.exists() else None
    )
    structural_validation_path = (
        None if args.auxiliary_dataset_dir is None
        else args.auxiliary_dataset_dir / "structural_validation.pt"
    )
    structural_validation = (
        load_world_split(
            args.auxiliary_dataset_dir, "structural_validation",
            args.max_eval_rows,
        )
        if structural_validation_path is not None
        and structural_validation_path.exists() else None
    )
    ensemble, histories, calibration, seconds = train_world_ensemble(
        train, validation, config, args.device, calibration_data,
        adaptation_data, structural_data, structural_validation,
    )
    artifact = save_world_ensemble(
        ensemble, histories, args.artifact_dir, dataset_manifest, calibration,
        fit_support_profile(train_sources, validation_sources),
    )
    return ensemble, artifact, seconds


def main():
    args = _parse_args()
    config = replace(
        WorldModelConfig(), epochs=args.epochs, batch_size=args.batch_size,
        ensemble_members=args.ensemble_members,
    )
    test = load_world_split(args.dataset_dir, "test", args.max_eval_rows)
    structural_test = (
        load_world_split(args.auxiliary_dataset_dir, "test", args.max_eval_rows)
        if args.auxiliary_dataset_dir is not None else None
    )
    primary_manifest = _load_dataset_manifest(args.dataset_dir)
    auxiliary_manifest = (
        _load_dataset_manifest(args.auxiliary_dataset_dir)
        if args.auxiliary_dataset_dir is not None else None
    )
    dataset_manifest = _combined_dataset_manifest(
        primary_manifest, auxiliary_manifest,
    )
    if args.reuse_artifact:
        ensemble, artifact_manifest, training_seconds = _reuse_artifact(
            args, dataset_manifest,
        )
    else:
        ensemble, artifact_manifest, training_seconds = _train_artifact(
            args, config, dataset_manifest,
        )
    evaluation = evaluate_world_model(
        ensemble, test, args.device, artifact_manifest["manifest_sha256"],
        args.policy_evidence, structural_split=structural_test,
        support_profile=artifact_manifest.get("support_profile"),
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
