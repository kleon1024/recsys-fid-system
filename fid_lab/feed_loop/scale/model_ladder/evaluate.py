"""Offline, GAUC, calibration, and candidate-order evaluation for V3 models."""

from __future__ import annotations

import numpy as np
import torch

from ....evolution.evaluation.metrics import binary_metrics, grouped_auc
from ...models.multitask_policy import FeedMMoEPolicy
from ...models.feed_multitask import FeedMultiTaskPolicy
from ...models.feed_multitask import TASK_SPECS
from .data import ExposedSplit
from .train import LONG_VIEW


def long_view_scores(model, features):
    if isinstance(model, (FeedMMoEPolicy, FeedMultiTaskPolicy)):
        return model.predict_tasks(features)["long_view"]
    return model.score(features)


def offline_metrics(model, split: ExposedSplit):
    scores = long_view_scores(model, split.features)
    labels = split.labels[:, LONG_VIEW]
    return {
        **binary_metrics(labels, scores),
        "user_gauc": grouped_auc(labels, scores, split.user_ids),
    }


def rule_offline_metrics(split: ExposedSplit):
    scores = 1.0 / (1.0 + np.exp(-split.rule_scores))
    labels = split.labels[:, LONG_VIEW]
    return {
        **binary_metrics(labels, scores),
        "user_gauc": grouped_auc(labels, scores, split.user_ids),
    }


def multitask_metrics(model: FeedMultiTaskPolicy, split: ExposedSplit):
    predictions = model.predict_tasks(split.features)
    report = {}
    for task, (kind, label_index, _) in TASK_SPECS.items():
        target = split.labels[:, label_index]
        score = predictions[task]
        if task == "stay_norm":
            target = target / 180.0
        if kind == "binary":
            metrics = binary_metrics(target, score)
            metrics["user_gauc"] = grouped_auc(target, score, split.user_ids)
        else:
            error = score - target
            metrics = {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
            }
        report[task] = metrics
    report["slices"] = {
        "lifecycle": _slice_long_view(predictions["long_view"], split, split.lifecycle),
        "region": _slice_long_view(predictions["long_view"], split, split.region),
    }
    return report


def _slice_long_view(scores, split, groups):
    labels = split.labels[:, LONG_VIEW]
    output = {}
    for group in np.unique(groups):
        mask = groups == group
        output[str(int(group))] = {
            "rows": int(mask.sum()),
            **binary_metrics(labels[mask], scores[mask]),
        }
    return output


def _score_candidates(model, features, chunk=200_000):
    flat = features.reshape(-1, features.shape[-1])
    output = []
    for start in range(0, len(flat), chunk):
        output.append(long_view_scores(model, flat[start : start + chunk]))
    return np.concatenate(output).reshape(features.shape[:2])


def candidate_metrics(model, tensors, limit=50_000):
    count = min(limit, len(tensors["request_id"]))
    features = tensors["candidate_features"][:count].float().numpy()
    scores = _score_candidates(model, features)
    choice = scores.argmax(axis=1)
    utility = tensors["candidate_audit_utility"][:count].float().numpy()
    rows = np.arange(count)
    regret = utility.max(axis=1) - utility[rows, choice]
    return {
        "requests": count,
        "audit_regret": float(regret.mean()),
        "audit_top1_rate": float((choice == utility.argmax(axis=1)).mean()),
    }


def rule_candidate_metrics(tensors, limit=50_000):
    count = min(limit, len(tensors["request_id"]))
    scores = tensors["candidate_fine_scores"][:count].float().numpy()
    utility = tensors["candidate_audit_utility"][:count].float().numpy()
    choice = scores.argmax(axis=1)
    rows = np.arange(count)
    regret = utility.max(axis=1) - utility[rows, choice]
    return {
        "requests": count,
        "audit_regret": float(regret.mean()),
        "audit_top1_rate": float((choice == utility.argmax(axis=1)).mean()),
    }
