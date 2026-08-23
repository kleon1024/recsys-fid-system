"""Unified equal-data model-evolution benchmark and model-card output."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier

from ...scale.contracts import FEED_TASKS, ScaleConfig
from ...scale.synthetic import build_scale_dataset, summarize_distribution
from .metrics import binary_metrics
from ..models.deepctr_adapter import DeepCTRModelAdapter, build_din_bundle, build_feature_bundle
from .retrieval_benchmark import run_retrieval_benchmark


PROFILE_IMPRESSIONS = {
    "ci": 20_000,
    "smoke": 100_000,
    "local": 1_000_000,
    "gpu": 10_000_000,
}


@dataclass(frozen=True)
class ModelResult:
    name: str
    library: str
    seed: int
    metrics: dict[str, float]
    parameters: int | None
    train_seconds: float
    milliseconds_per_1k_predictions: float
    training_loss: tuple[float, ...]
    training_device: str
    prediction_device: str


def _matrix(sparse: np.ndarray, dense: np.ndarray) -> np.ndarray:
    normalized_sparse = (sparse % 128).astype(np.float32) / 127.0
    return np.concatenate([normalized_sparse, dense], axis=1)


def _split_inputs(inputs: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {name: values[start:stop] for name, values in inputs.items()}


def _traditional_models(
    sparse: np.ndarray,
    dense: np.ndarray,
    labels: np.ndarray,
    seed: int,
    device: str,
) -> list[ModelResult]:
    train_end = int(len(labels) * 0.70)
    test_start = int(len(labels) * 0.85)
    features = _matrix(sparse, dense)
    target = labels[:, 0]
    models = {
        "logistic_regression": LogisticRegression(max_iter=200, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            device=device,
            n_jobs=4,
            random_state=seed,
        ),
    }
    results = []
    for name, model in models.items():
        started = perf_counter()
        model.fit(features[:train_end], target[:train_end])
        train_seconds = perf_counter() - started
        training_device = "cpu" if name == "logistic_regression" else device
        if name == "xgboost" and device.startswith("cuda"):
            model.set_params(device="cpu", n_jobs=1)
        train_scores = model.predict_proba(features[:train_end])[:, 1]
        train_loss = binary_metrics(target[:train_end], train_scores)["log_loss"]
        started = perf_counter()
        scores = model.predict_proba(features[test_start:])[:, 1]
        prediction_ms = (perf_counter() - started) * 1_000_000.0 / len(scores)
        results.append(
            ModelResult(
                name,
                "scikit-learn" if name == "logistic_regression" else "xgboost",
                seed,
                binary_metrics(target[test_start:], scores),
                None,
                train_seconds,
                prediction_ms,
                (train_loss,),
                training_device,
                "cpu",
            )
        )
    return results


def _deepctr_models(
    sparse: np.ndarray,
    dense: np.ndarray,
    history_item_ids: np.ndarray,
    sequence_mask: np.ndarray,
    served_scores: np.ndarray,
    labels: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
) -> list[ModelResult]:
    train_end = int(len(labels) * 0.70)
    validation_end = int(len(labels) * 0.85)
    test_start = int(len(labels) * 0.85)
    bundle = build_feature_bundle(sparse, dense)
    din_bundle = build_din_bundle(sparse, dense, history_item_ids, sequence_mask)
    results = []
    for name in ("wide_deep", "deepfm", "dcnv2", "din", "mmoe", "ple"):
        model_bundle = din_bundle if name in {"din", "mmoe", "ple"} else bundle
        train_inputs = _split_inputs(model_bundle.inputs, 0, train_end)
        validation_inputs = _split_inputs(
            model_bundle.inputs, train_end, validation_end
        )
        test_inputs = _split_inputs(model_bundle.inputs, test_start, len(labels))
        tasks = FEED_TASKS[:3] if name in {"mmoe", "ple"} else ("long_view",)
        model = DeepCTRModelAdapter(name, model_bundle, tasks, device=device, seed=seed)
        target = labels[:train_end, : len(tasks)]
        if len(tasks) == 1:
            target = target[:, 0]
        validation_target = labels[train_end:validation_end, : len(tasks)]
        if len(tasks) == 1:
            validation_target = validation_target[:, 0]
        started = perf_counter()
        model.fit(
            train_inputs,
            target,
            epochs=epochs,
            validation=(validation_inputs, validation_target),
        )
        if name == "dcnv2":
            teacher = served_scores[:train_end, 2]
            ranks = np.argsort(np.argsort(-teacher))
            model.fit_distilled(
                train_inputs,
                labels[:train_end, 0],
                teacher,
                ranks,
                epochs=min(epochs, 3),
            )
        train_seconds = perf_counter() - started
        started = perf_counter()
        scores = model.predict(test_inputs)[:, 0]
        prediction_ms = (perf_counter() - started) * 1_000_000.0 / len(scores)
        results.append(
            ModelResult(
                f"{name}_distilled" if name == "dcnv2" else name,
                "deepctr-torch-0.3.0",
                seed,
                binary_metrics(labels[test_start:, 0], scores),
                model.parameters,
                train_seconds,
                prediction_ms,
                tuple(model.loss_history),
                device,
                device,
            )
        )
    return results


def run_benchmark(
    profile: str = "smoke",
    seeds: int = 1,
    device: str = "cpu",
    epochs: int = 1,
    signal_version: str = "industrial-cross-sequence-v1",
) -> dict[str, object]:
    if profile not in PROFILE_IMPRESSIONS:
        raise ValueError(f"unknown profile: {profile}")
    torch.set_num_threads(4)
    reports = []
    distribution = None
    for offset in range(seeds):
        seed = 20260823 + offset
        dataset = build_scale_dataset(
            ScaleConfig(
                main_impressions=PROFILE_IMPRESSIONS[profile],
                seed=seed,
                signal_version=signal_version,
            )
        )
        distribution = summarize_distribution(dataset)
        reports.extend(
            _traditional_models(
                dataset.sparse_ids,
                dataset.dense_features,
                dataset.labels,
                seed,
                device,
            )
        )
        reports.extend(
            _deepctr_models(
                dataset.sparse_ids,
                dataset.dense_features,
                dataset.history_item_ids,
                dataset.sequence_mask,
                dataset.served_scores,
                dataset.labels,
                seed,
                device,
                epochs,
            )
        )
    return {
        "profile": profile,
        "seeds": seeds,
        "signal_version": signal_version,
        "distribution": distribution,
        "ranking": [asdict(report) for report in reports],
        "retrieval": run_retrieval_benchmark(
            items=500 if profile == "ci" else (2_000 if profile == "smoke" else 5_000),
            queries=200 if profile == "ci" else (800 if profile == "smoke" else 2_000),
            device=device,
        ),
    }


def to_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Model evolution leaderboard",
        "",
        "| Model | Library | AUC | PR-AUC | LogLoss | ECE | Final train loss | Train seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["ranking"]:
        metric = result["metrics"]
        lines.append(
            f"| {result['name']} | {result['library']} | {metric['auc']:.4f} | "
            f"{metric['pr_auc']:.4f} | {metric['log_loss']:.4f} | {metric['ece']:.4f} | "
            f"{result['training_loss'][-1]:.4f} | "
            f"{result['train_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILE_IMPRESSIONS), default="ci")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--signal-version",
        default="industrial-cross-sequence-v1",
        choices=("industrial-cross-sequence-v1", "heterogeneous-nonlinear-v2"),
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    report = run_benchmark(
        args.profile,
        args.seeds,
        args.device,
        args.epochs,
        args.signal_version,
    )
    print(to_markdown(report) if args.format == "markdown" else json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
