"""Train external W&D and sequence kernels and publish falsification evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from xgboost import XGBClassifier

from ..data.sequence import load_sequence_split
from ..launch.contracts import stream_sha256
from .architectures import KuaiSequenceTransformer, KuaiWideDeep
from .training import behavior_metrics, fit_behavior_model, predict_behavior


def tabular_matrix(split, vocabularies):
    sparse = split.sparse.numpy().astype(np.float32)
    sparse /= np.asarray(vocabularies, dtype=np.float32)[None]
    return np.concatenate((sparse, split.dense.numpy()), axis=1)


def fit_tabular(train, validation, vocabularies, device, seed):
    train_x = tabular_matrix(train, vocabularies)
    validation_x = tabular_matrix(validation, vocabularies)
    train_y = train.labels[:, 1].numpy()
    validation_y = validation.labels[:, 1].numpy()
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=seed),
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            device=device.type, random_state=seed, n_jobs=4,
        ),
    }
    timing = {}
    for name, model in models.items():
        started = perf_counter()
        if name == "xgboost":
            model.fit(
                train_x, train_y, eval_set=[(validation_x, validation_y)],
                verbose=False,
            )
        else:
            model.fit(train_x, train_y)
        timing[name] = perf_counter() - started
    return models, timing


def tabular_metrics(models, timing, test, vocabularies):
    features = tabular_matrix(test, vocabularies)
    labels = test.labels[:, 1].numpy()
    return {
        name: {
            "long_view_auc": float(roc_auc_score(labels, model.predict_proba(features)[:, 1])),
            "long_view_log_loss": float(log_loss(labels, model.predict_proba(features)[:, 1])),
            "train_seconds": timing[name],
        }
        for name, model in models.items()
    }


def _save_model(model, path, metadata):
    torch.save({"state_dict": model.state_dict(), **metadata}, path)
    return stream_sha256(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    train = load_sequence_split(args.dataset_dir, "train", args.max_train_rows)
    validation = load_sequence_split(
        args.dataset_dir, "validation", args.max_eval_rows
    )
    test = load_sequence_split(args.dataset_dir, "test", args.max_eval_rows)
    device = torch.device(args.device)
    vocabularies = tuple(manifest["sparse_vocabularies"])
    tabular, timing = fit_tabular(
        train, validation, vocabularies, device, args.seed
    )
    results = tabular_metrics(tabular, timing, test, vocabularies)
    torch.manual_seed(args.seed)
    models = {
        "wide_deep": KuaiWideDeep(vocabularies, train.dense.shape[1]),
        "sequence_transformer": KuaiSequenceTransformer(
            vocabularies, train.dense.shape[1], train.history_items.shape[1]
        ),
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        history, seconds = fit_behavior_model(
            model, train, validation, device, args.epochs, args.seed
        )
        predictions = predict_behavior(model, test, device)
        results[name] = {
            "tasks": behavior_metrics(test.labels.numpy(), predictions),
            "long_view_auc": float(
                roc_auc_score(test.labels[:, 1].numpy(), predictions[:, 1])
            ),
            "train_seconds": seconds,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_history": history,
        }
        results[name]["artifact_sha256"] = _save_model(
            model, args.artifact_dir / f"{name}.pt",
            {"dataset_manifest": manifest, "model_name": name},
        )
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(test), generator=generator)
    sequence = models["sequence_transformer"]
    baseline = predict_behavior(sequence, test, device)
    shuffled = predict_behavior(sequence, test, device, history_permutation=permutation)
    sequence_delta = np.abs(baseline[:, 1] - shuffled[:, 1])
    report = {
        "schema": "kuairand-external-capacity-benchmark-v1",
        "seed": args.seed,
        "dataset_manifest": manifest,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "models": results,
        "history_permutation": {
            "mean_absolute_long_view_delta": float(sequence_delta.mean()),
            "p95_absolute_long_view_delta": float(np.quantile(sequence_delta, 0.95)),
            "shuffled_long_view_auc": float(
                roc_auc_score(test.labels[:, 1].numpy(), shuffled[:, 1])
            ),
        },
    }
    report["gates"] = {
        "wide_deep_beats_logistic": (
            results["wide_deep"]["long_view_auc"]
            > results["logistic_regression"]["long_view_auc"]
        ),
        "sequence_beats_best_tabular": (
            results["sequence_transformer"]["long_view_auc"]
            > max(results["xgboost"]["long_view_auc"], results["wide_deep"]["long_view_auc"])
        ),
        "history_is_material": float(sequence_delta.mean()) >= 0.005,
    }
    report["decision"] = (
        "external_sequence_capacity_pass" if all(report["gates"].values())
        else "hold_external_sequence_capacity"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "long_view_auc": {
            name: evidence["long_view_auc"] for name, evidence in results.items()
        },
        "history_permutation": report["history_permutation"],
    }, indent=2))


if __name__ == "__main__":
    main()
