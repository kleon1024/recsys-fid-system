"""GPU benchmark against untouched KuaiRand randomized exposures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .sequence_benchmark import fit_tabular, tabular_metrics
from ..evaluation.calibration import randomized_calibration
from ..contracts import FEEDBACK_NAMES
from .architectures import KuaiSequenceMMoE, KuaiSequenceTransformer, KuaiWideDeep
from ..data.randomized import load_randomized_split
from ..launch.contracts import stream_sha256
from .training import behavior_metrics, fit_behavior_model, predict_behavior


METRIC_NAMES = (*FEEDBACK_NAMES, "stay_norm")


def _save_model(model, path, manifest, architecture, seed):
    torch.save({
        "state_dict": model.state_dict(), "dataset_manifest": manifest,
        "model_name": architecture, "seed": seed,
    }, path)
    return stream_sha256(path)


def _fit_neural(name, architecture, model, train, validation, tests, device,
                epochs, seed, artifact_dir, manifest):
    history, seconds = fit_behavior_model(
        model, train, validation, device, epochs, seed
    )
    evaluations = {}
    predictions = {}
    for split_name, split in tests.items():
        prediction = predict_behavior(model, split, device)
        predictions[split_name] = prediction
        evaluations[split_name] = behavior_metrics(
            split.labels.numpy(), prediction
        )
    artifact_path = artifact_dir / f"{name}.pt"
    return {
        "train_seconds": seconds,
        "parameters": sum(value.numel() for value in model.parameters()),
        "training_history": history,
        "evaluations": evaluations,
        "artifact_sha256": _save_model(
            model, artifact_path, manifest, architecture, seed
        ),
    }, predictions


def _grouped_effect(standard, random, values, seed=20260824, bootstraps=500):
    import pandas as pd

    columns = [*METRIC_NAMES]
    standard_frame = pd.DataFrame(values["standard"], columns=columns)
    random_frame = pd.DataFrame(values["random"], columns=columns)
    for frame, split in ((standard_frame, standard), (random_frame, random)):
        frame["user_id"] = split.user_ids.numpy()
        frame["date"] = split.dates.numpy()
    keys = ["user_id", "date"]
    standard_group = standard_frame.groupby(keys)[columns].mean()
    random_group = random_frame.groupby(keys)[columns].mean()
    paired = random_group.join(
        standard_group, how="inner", lsuffix="_random", rsuffix="_standard"
    ).dropna()
    if paired.empty:
        raise ValueError("no user-date support overlaps randomized and standard logs")
    effects = np.stack([
        paired[f"{name}_random"] - paired[f"{name}_standard"]
        for name in columns
    ], axis=1)
    users = paired.index.get_level_values("user_id").to_numpy()
    unique = np.unique(users)
    rng = np.random.default_rng(seed)
    samples = np.empty((bootstraps, len(columns)), dtype=float)
    by_user = {user: effects[users == user] for user in unique}
    for index in range(bootstraps):
        selected = rng.choice(unique, len(unique), replace=True)
        samples[index] = np.concatenate([by_user[user] for user in selected]).mean(axis=0)
    return {
        name: {
            "absolute_effect": float(effects[:, index].mean()),
            "confidence_interval_95": np.quantile(
                samples[:, index], (0.025, 0.975)
            ).tolist(),
        }
        for index, name in enumerate(columns)
    }


def _observed_values(split):
    return split.labels.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    splits = {
        name: load_randomized_split(args.dataset_dir, name)
        for name in ("train", "validation", "standard_test", "random_test")
    }
    train, validation = splits["train"], splits["validation"]
    tests = {name: splits[name] for name in ("standard_test", "random_test")}
    device = torch.device(args.device)
    vocabularies = tuple(manifest["sparse_vocabularies"])
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tabular, timing = fit_tabular(
        train, validation, vocabularies, device, args.seed
    )
    tabular_results = {
        name: tabular_metrics(tabular, timing, split, vocabularies)
        for name, split in tests.items()
    }
    model_specs = (
        ("wide_deep", "wide_deep", KuaiWideDeep),
        ("sequence_transformer", "sequence_transformer", KuaiSequenceTransformer),
        ("sequence_mmoe", "sequence_mmoe", KuaiSequenceMMoE),
        ("independent_world", "sequence_transformer", KuaiSequenceTransformer),
    )
    evidence, predictions = {}, {}
    for offset, (name, architecture, model_type) in enumerate(model_specs):
        model_seed = args.seed + offset
        torch.manual_seed(model_seed)
        if model_type is KuaiWideDeep:
            model = model_type(vocabularies, train.dense.shape[1])
        else:
            model = model_type(
                vocabularies, train.dense.shape[1], train.history_items.shape[1]
            )
        evidence[name], predictions[name] = _fit_neural(
            name, architecture, model, train, validation, tests, device,
            args.epochs, model_seed, args.artifact_dir, manifest,
        )
    calibrations = {
        name: randomized_calibration(
            splits["random_test"], result["random_test"], args.seed
        )
        for name, result in predictions.items()
    }
    effects = _grouped_effect(
        splits["standard_test"], splits["random_test"], {
            "standard": _observed_values(splits["standard_test"]),
            "random": _observed_values(splits["random_test"]),
        }, args.seed,
    )
    random_auc = {
        name: value["evaluations"]["random_test"]["long_view"]["auc"]
        for name, value in evidence.items()
    }
    gates = {
        "sequence_beats_wide_deep_randomized": (
            random_auc["sequence_transformer"] > random_auc["wide_deep"]
        ),
        "random_intervention_reduces_long_view": (
            effects["long_view"]["confidence_interval_95"][1] < 0
        ),
        "random_intervention_reduces_stay": (
            effects["stay_norm"]["confidence_interval_95"][1] < 0
        ),
        "randomized_hate_has_positive_support": (
            calibrations["independent_world"]["is_hate"]["evaluation_positives"] > 0
        ),
        "policy_order_from_randomized_ope": False,
    }
    report = {
        "schema": "kuairand-1k-randomized-capacity-v1",
        "seed": args.seed, "dataset_manifest": manifest,
        "rows": {name: len(split) for name, split in splits.items()},
        "tabular_models": tabular_results,
        "neural_models": evidence,
        "randomized_calibration": calibrations,
        "random_intervention_effect": effects,
        "gates": gates,
        "decision": "hold_randomized_ope_pending",
        "evidence_boundary": (
            "Random intervention recovery and randomized generalization are "
            "measured; policy ordering remains closed until DR/OPE is executed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": gates,
        "random_long_view_auc": random_auc,
        "intervention_effect": effects,
    }, indent=2))


if __name__ == "__main__":
    main()
