"""Equal-data V4 Feed capacity benchmark and artifact publication."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from xgboost import XGBClassifier

from .....evolution.evaluation.metrics import binary_metrics, grouped_auc
from .....evolution.models.deepctr_adapter import DeepCTRModelAdapter
from ....world_model.benchmark.data import deepctr_bundle, score_tabular_model
from .contracts import TASKS
from .data import load_request_split
from .evaluation import evaluate_request_model, sequence_ablation
from .networks import MMoERanker, PLERanker, SingleTaskDIN, SingleTaskTransformer
from .training import fit_request_model


LONG_VIEW_LABEL = 5


def _tabular_models(train, validation, device, epochs, seed):
    train_x = train.selected_features()
    train_y = train.labels()[:, LONG_VIEW_LABEL]
    validation_x = validation.selected_features()
    validation_y = validation.labels()[:, LONG_VIEW_LABEL]
    weights = train.weights()
    models = {}
    started = perf_counter()
    lr = LogisticRegression(max_iter=400, random_state=seed)
    lr.fit(train_x, train_y, sample_weight=weights)
    models["lr"] = (lr, perf_counter() - started, ())
    started = perf_counter()
    xgboost = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=20,
        reg_lambda=2.0, tree_method="hist", device=device.type,
        random_state=seed, n_jobs=4,
    )
    xgboost.fit(
        train_x, train_y, sample_weight=weights,
        eval_set=[(validation_x, validation_y)], verbose=False,
    )
    xgboost.set_params(device="cpu", n_jobs=4)
    models["xgboost"] = (xgboost, perf_counter() - started, ())
    probability = weights / weights.sum()
    sampled = np.random.default_rng(seed).choice(
        len(train_x), len(train_x), replace=True, p=probability
    )
    train_bundle = deepctr_bundle(train_x[sampled])
    validation_bundle = deepctr_bundle(validation_x)
    for name in ("wide_deep", "deepfm", "dcnv2"):
        model = DeepCTRModelAdapter(
            name, train_bundle, device=str(device), seed=seed
        )
        started = perf_counter()
        model.fit(
            train_bundle.inputs, train_y[sampled], epochs=epochs,
            validation=(validation_bundle.inputs, validation_y),
        )
        models[name] = (model, perf_counter() - started, tuple(model.loss_history))
    return models


def _evaluate_tabular(model, split, ranking_rows):
    features = split.selected_features()
    labels = split.labels()[:, LONG_VIEW_LABEL]
    scores = score_tabular_model(model, features)
    offline = binary_metrics(labels, scores)
    offline["user_gauc"] = grouped_auc(labels, scores, split.user_ids())
    count = min(ranking_rows, len(split))
    candidate = split.tensors["candidate_features"][
        split.indices[:count]
    ].float().numpy()
    flat = candidate.reshape(-1, candidate.shape[-1])
    parts = []
    for start in range(0, len(flat), 250_000):
        parts.append(score_tabular_model(model, flat[start : start + 250_000]))
    candidate_scores = np.concatenate(parts).reshape(candidate.shape[:2])
    utility = split.tensors["candidate_audit_utility"][
        split.indices[:count]
    ].float().numpy()
    choice = candidate_scores.argmax(axis=1)
    rows = np.arange(count)
    return {
        "offline": offline,
        "heads": {},
        "ranking": {
            "requests": count,
            "audit_regret": float(
                (utility.max(axis=1) - utility[rows, choice]).mean()
            ),
            "audit_top1_rate": float(
                (choice == utility.argmax(axis=1)).mean()
            ),
        },
    }


def _publish(name, model, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if name == "lr":
        path = artifact_dir / "lr_v4.joblib"
        joblib.dump(model, path)
    elif name == "xgboost":
        path = artifact_dir / "xgboost_v4.json"
        model.save_model(path)
    else:
        path = artifact_dir / f"{name}_v4.pt"
        target = model.model if isinstance(model, DeepCTRModelAdapter) else model
        torch.save(target.state_dict(), path)
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _rule_metrics(test, ranking_rows):
    index = test.indices
    exposed = test.tensors["exposed_index"][index].long()
    rows = torch.arange(len(index))
    raw = test.tensors["candidate_fine_scores"][index].float()
    selected = torch.sigmoid(raw[rows, exposed]).numpy()
    labels = test.labels()[:, LONG_VIEW_LABEL]
    offline = binary_metrics(labels, selected)
    offline["user_gauc"] = grouped_auc(labels, selected, test.user_ids())
    count = min(ranking_rows, len(test))
    score = raw[:count].numpy()
    utility = test.tensors["candidate_audit_utility"][index[:count]].float().numpy()
    choice = score.argmax(axis=1)
    request_rows = np.arange(count)
    return {
        "offline": offline,
        "heads": {},
        "ranking": {
            "requests": count,
            "audit_regret": float(
                (utility.max(axis=1) - utility[request_rows, choice]).mean()
            ),
            "audit_top1_rate": float((choice == utility.argmax(axis=1)).mean()),
        },
    }


def run_request_ladder(
    dataset_dir: Path,
    artifact_dir: Path,
    device_name="cuda:0",
    train_rows=200_000,
    validation_rows=60_000,
    test_rows=60_000,
    ranking_rows=20_000,
    epochs=5,
    seed=20260823,
):
    device = torch.device(device_name)
    train = load_request_split(dataset_dir, "train", train_rows, seed)
    validation = load_request_split(dataset_dir, "validation", validation_rows, seed + 1)
    test = load_request_split(dataset_dir, "test", test_rows, seed + 2)
    results = {
        "rule_personalized": {
            **_rule_metrics(test, ranking_rows),
            "information_view": "served_rule_features",
            "training_seconds": 0.0, "parameters": 0, "artifact": None,
        }
    }
    for name, (model, seconds, history) in _tabular_models(
        train, validation, device, epochs, seed
    ).items():
        results[name] = {
            **_evaluate_tabular(model, test, ranking_rows),
            "information_view": "selected_candidate_snapshot_only",
            "training_seconds": seconds,
            "parameters": (
                model.parameters if isinstance(model, DeepCTRModelAdapter) else None
            ),
            "training_history": history,
            "artifact": _publish(name, model, artifact_dir),
        }
    staged_at = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        train.stage(device)
        validation.stage(device)
        test.stage(device)
    staging_seconds = perf_counter() - staged_at
    request_models = (
        ("din", SingleTaskDIN(train.feature_dim, train.sequence_dim)),
        ("transformer", SingleTaskTransformer(train.feature_dim, train.sequence_dim)),
        ("mmoe", MMoERanker(len(TASKS), train.feature_dim, train.sequence_dim)),
        ("ple", PLERanker(len(TASKS), train.feature_dim, train.sequence_dim)),
    )
    for name, model in request_models:
        started = perf_counter()
        history = fit_request_model(model, train, validation, device, epochs, seed)
        results[name] = {
            **evaluate_request_model(model, test, device, ranking_rows),
            "information_view": (
                "candidate_plus_sequence" if name == "din"
                else "candidate_plus_sequence_plus_slate" if name == "transformer"
                else "candidate_plus_sequence_multi_task"
            ),
            "training_seconds": perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_history": history,
            "sequence_ablation": sequence_ablation(
                model, test, device, min(ranking_rows, 10_000), seed
            ),
            "artifact": _publish(name, model, artifact_dir),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "schema": "v4-request-aware-model-ladder-v1",
        "dataset_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "behavior_world": manifest["behavior_world"],
        "contract": {
            "train_rows": len(train), "validation_rows": len(validation),
            "test_rows": len(test), "ranking_rows": min(ranking_rows, len(test)),
            "candidate_count": train.candidate_count,
            "sequence_length": train.tensors["behavior_sequence"].shape[1],
            "tasks": [task.name for task in TASKS],
            "candidate_capacity_metric": (
                "predicted primitive heads combined with the frozen DGP audit weights"
            ),
            "product_value_tree_is_a_separate_ab_policy": True,
            "same_frozen_splits": True, "same_candidate_set": True,
            "ips_clip": train.ips_clip, "lt_is_not_a_training_label": True,
            "request_tensors_gpu_resident": device.type == "cuda",
            "gpu_staging_seconds": staging_seconds,
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else None
            ),
        },
        "models": results,
        "evidence_boundary": "Synthetic V4 offline evidence; A/B is a separate gate.",
    }
