"""Run an equal-data V4 capacity ladder without tuning the DGP to a winner."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
import torch
from xgboost import XGBClassifier

from ....evolution.models.deepctr_adapter import DeepCTRModelAdapter
from ..data import load_world_split
from ..training import load_world_ensemble
from .data import (
    candidate_oracle,
    context_ablation,
    deepctr_bundle,
    information_ceiling,
    materialize_target,
    score_tabular_model,
)
from .contracts import capacity_gates
from .neural import (
    DINRequestRanker,
    SlateTransformerRanker,
    fit_request_ranker,
    predict_request_ranker,
)


def _exposed_metrics(labels, scores, oracle, seconds, parameters=None, history=()):
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return {
        "auc": float(roc_auc_score(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped)),
        "oracle_probability_mse": float(np.mean(np.square(clipped - oracle))),
        "train_seconds": seconds,
        "parameters": parameters,
        "training_history": history,
    }


def _candidate_metrics(scores, oracle):
    choice = scores.argmax(axis=1)
    rows = np.arange(len(scores))
    oracle_choice = oracle.argmax(axis=1)
    return {
        "oracle_regret": float(np.mean(oracle.max(axis=1) - oracle[rows, choice])),
        "top1_agreement": float(np.mean(choice == oracle_choice)),
        "ndcg_at_10": float(ndcg_score(oracle, scores, k=10)),
    }


def _fit_tabular(train, train_target, validation, validation_target, seed, device):
    models = {}
    started = perf_counter()
    logistic = LogisticRegression(max_iter=400, random_state=seed)
    logistic.fit(train.selected_features.numpy(), train_target.labels)
    models["logistic_regression"] = (logistic, perf_counter() - started)
    started = perf_counter()
    xgboost = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
        reg_lambda=2.0, tree_method="hist", device=device.type,
        random_state=seed, n_jobs=4,
    )
    xgboost.fit(
        train.selected_features.numpy(), train_target.labels,
        eval_set=[(validation.selected_features.numpy(), validation_target.labels)],
        verbose=False,
    )
    models["xgboost"] = (xgboost, perf_counter() - started)
    return models


def _fit_deepctr(train, train_target, validation, validation_target,
                 seed, device, epochs):
    train_bundle = deepctr_bundle(train.selected_features.numpy())
    validation_bundle = deepctr_bundle(validation.selected_features.numpy())
    models = {}
    for name in ("wide_deep", "deepfm", "dcnv2"):
        model = DeepCTRModelAdapter(
            name, train_bundle, device=str(device), seed=seed
        )
        started = perf_counter()
        model.fit(
            train_bundle.inputs, train_target.labels, epochs=epochs,
            validation=(validation_bundle.inputs, validation_target.labels),
        )
        models[name] = (model, perf_counter() - started)
    return models


def _evaluate_tabular(models, test, target, candidate_features, oracle_candidates):
    results = {}
    flat_candidates = candidate_features.reshape(-1, candidate_features.shape[-1])
    for name, (model, seconds) in models.items():
        exposed = score_tabular_model(model, test.selected_features.numpy())
        candidates = score_tabular_model(model, flat_candidates).reshape(
            candidate_features.shape[:2]
        )
        parameters = model.parameters if isinstance(model, DeepCTRModelAdapter) else None
        history = tuple(model.loss_history) if isinstance(model, DeepCTRModelAdapter) else ()
        results[name] = {
            "information_view": "selected_sparse_dense_only",
            **_exposed_metrics(
                target.labels, exposed, target.oracle_probability,
                seconds, parameters, history,
            ),
            "request": _candidate_metrics(candidates, oracle_candidates),
        }
    return results


def _fit_request_models(train, train_target, validation, validation_target,
                        device, epochs, seed):
    output = {}
    for name, model in (
        ("din_request", DINRequestRanker()),
        ("slate_transformer", SlateTransformerRanker()),
    ):
        started = perf_counter()
        history = fit_request_ranker(
            model, train, train_target.labels, validation,
            validation_target.labels, device, epochs, seed,
        )
        output[name] = (model, perf_counter() - started, history)
    return output


def run_model_capacity_benchmark(dataset_dir: Path, artifact_dir: Path,
                                 device_name="cuda:0", train_rows=300_000,
                                 eval_rows=100_000, request_rows=10_000,
                                 epochs=8, seed=20260823):
    device = torch.device(device_name)
    ensemble = load_world_ensemble(artifact_dir, device_name).eval()
    train = load_world_split(dataset_dir, "train", train_rows, "uniform")
    validation = load_world_split(
        dataset_dir, "validation", eval_rows, "uniform"
    )
    test = load_world_split(dataset_dir, "test", eval_rows, "uniform")
    targets = {
        "train": materialize_target(ensemble, train, device, seed + 1),
        "validation": materialize_target(ensemble, validation, device, seed + 2),
        "test": materialize_target(ensemble, test, device, seed + 3),
    }
    oracle_candidates = candidate_oracle(
        ensemble, test, device, request_rows
    )
    candidate_features = test.slate_features[:request_rows].numpy()
    tabular = _fit_tabular(
        train, targets["train"], validation, targets["validation"], seed, device
    )
    tabular.update(_fit_deepctr(
        train, targets["train"], validation, targets["validation"],
        seed, device, epochs,
    ))
    results = _evaluate_tabular(
        tabular, test, targets["test"], candidate_features, oracle_candidates
    )
    request_models = _fit_request_models(
        train, targets["train"], validation, targets["validation"],
        device, epochs, seed,
    )
    for name, (model, seconds, history) in request_models.items():
        exposed_matrix = predict_request_ranker(model, test, device)
        rows = np.arange(len(test))
        exposed = exposed_matrix[rows, test.exposed_index.numpy()]
        candidate_scores = exposed_matrix[:request_rows]
        results[name] = {
            "information_view": (
                "selected_plus_sequence" if name == "din_request"
                else "selected_plus_sequence_plus_slate"
            ),
            **_exposed_metrics(
                targets["test"].labels, exposed,
                targets["test"].oracle_probability, seconds,
                sum(parameter.numel() for parameter in model.parameters()), history,
            ),
            "request": _candidate_metrics(candidate_scores, oracle_candidates),
        }
    context = context_ablation(
        ensemble, test, device, min(eval_rows, 50_000), seed
    )
    gates = capacity_gates(context, results)
    return {
        "schema": "v4-request-model-capacity-benchmark-v1",
        "contract": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "request_metric_rows": min(request_rows, len(test)),
            "target": "sampled_v4_long_view",
            "oracle": "v4_ensemble_mean_long_view_probability",
            "same_split_and_candidate_budget": True,
            "row_selection": "uniform_across_each_frozen_time_split",
            "request_step_support": {
                "train": sorted(set(train.request_steps.tolist())),
                "validation": sorted(set(validation.request_steps.tolist())),
                "test": sorted(set(test.request_steps.tolist())),
            },
        },
        "information_ceiling": information_ceiling(
            targets["test"].labels, targets["test"].oracle_probability
        ),
        "context_ablation": context,
        "models": results,
        "gates": gates,
        "decision": (
            "capacity_separation_pass" if all(gates.values())
            else "hold_v4_sequence_capacity_not_proven"
        ),
    }
