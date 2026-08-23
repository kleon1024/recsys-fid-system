"""Fine-tune the sequence ranker on randomized calibration users only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..evaluation.calibration import randomized_calibration
from ..kernel import KuaiBehaviorKernel
from ..launch.contracts import assert_artifact_compatible, stream_sha256
from .pairwise import fit_pairwise_randomized, mine_user_day_pairs
from ..data.randomized import calibration_masks, load_randomized_split, subset_split
from .training import behavior_metrics, fit_behavior_model, predict_behavior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--parent-artifact", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    assert_artifact_compatible(args.dataset_dir, (args.parent_artifact,))
    split = load_randomized_split(args.dataset_dir, "random_test")
    calibration, evaluation = calibration_masks(split, args.seed)
    calibration_users = split.user_ids[np.flatnonzero(calibration)].numpy()
    validation = calibration & (split.user_ids.numpy() % 5 == 0)
    training = calibration & ~validation
    training_index = np.flatnonzero(training)
    validation_index = np.flatnonzero(validation)
    evaluation_index = np.flatnonzero(evaluation)
    if (
        split.labels[training_index, 1].sum() < 100
        or split.labels[validation_index, 1].sum() < 20
    ):
        raise ValueError("randomized adapter lacks long-view support")
    kernel = KuaiBehaviorKernel.load(args.parent_artifact, args.device)
    history, seconds = fit_behavior_model(
        kernel.model, subset_split(split, training_index),
        subset_split(split, validation_index), torch.device(args.device),
        args.epochs, args.seed + 101, batch_size=1_024, learning_rate=1e-4,
    )
    pointwise_predictions = predict_behavior(
        kernel.model, split, torch.device(args.device)
    )
    pairs = mine_user_day_pairs(split, training, pointwise_predictions)
    pairwise_history = fit_pairwise_randomized(
        kernel.model, split, pairs, torch.device(args.device), args.seed + 202
    )
    predictions = predict_behavior(kernel.model, split, torch.device(args.device))
    metrics = behavior_metrics(
        split.labels[evaluation_index].numpy(), predictions[evaluation]
    )
    calibration_report = randomized_calibration(split, predictions, args.seed)
    parent_report = json.loads(args.benchmark_report.read_text())
    baseline_auc = parent_report["neural_models"]["sequence_transformer"][
        "evaluations"
    ]["random_test"]["long_view"]["auc"]
    payload = torch.load(
        args.parent_artifact, map_location="cpu", weights_only=False
    )
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": kernel.model.cpu().state_dict(),
        "dataset_manifest": payload["dataset_manifest"],
        "model_name": payload["model_name"],
        "seed": args.seed + 101,
        "parent_artifact_sha256": stream_sha256(args.parent_artifact),
        "adaptation": "randomized-calibration-users-v1",
    }, args.output_artifact)
    gates = {
        "user_disjoint_evaluation": not np.isin(
            split.user_ids[evaluation_index].numpy(), np.unique(calibration_users)
        ).any(),
        "randomized_long_view_auc_improves": metrics["long_view"]["auc"] > baseline_auc,
    }
    report = {
        "schema": "kuairand-randomized-sequence-adapter-v1",
        "seed": args.seed, "rows": {
            "train": int(training.sum()), "validation": int(validation.sum()),
            "evaluation": int(evaluation.sum()),
        },
        "training_history": history, "train_seconds": seconds,
        "pairwise": {
            "pairs": len(pairs), "loss": pairwise_history,
            "scope": "same-user same-day random exposures; not a request slate",
        },
        "evaluation": metrics,
        "randomized_calibration": {
            "sequence_randomized_adapter": calibration_report
        },
        "parent_artifact_sha256": stream_sha256(args.parent_artifact),
        "artifact_sha256": stream_sha256(args.output_artifact),
        "gates": gates,
        "decision": "adapter_candidate" if all(gates.values()) else "adapter_reject",
        "evidence_boundary": (
            "Fine-tuned only on calibration users; policy value is decided by "
            "randomized OPE on the disjoint evaluation users."
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": gates,
        "long_view": metrics["long_view"], "rows": report["rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
