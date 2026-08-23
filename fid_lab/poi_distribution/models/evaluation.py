"""Calibration and offline metrics for trained POI distribution rankers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from ..contracts import TASK_LABELS
from .objectives import targets


def metrics(bundle, split, limit=250_000):
    rows = min(len(split), limit)
    features = split.features[:rows].to(bundle.mean.device)
    predictions = {
        task: value.cpu().numpy()
        for task, value in bundle.probabilities(features).items()
    }
    labels = split.labels[:rows].numpy()
    report = {}
    for task, index in TASK_LABELS.items():
        target = labels[:, index]
        if task == "stay_norm":
            report[task] = {
                "mae": float(np.abs(predictions[task] - target / 180.0).mean()),
                "mean": float(target.mean() / 180.0),
            }
            continue
        report[task] = {
            "auc": (
                float(roc_auc_score(target, predictions[task]))
                if np.unique(target).size == 2
                else None
            ),
            "pr_auc": (
                float(average_precision_score(target, predictions[task]))
                if target.sum()
                else None
            ),
            "positive_rate": float(target.mean()),
            "positive_rows": int(target.sum()),
        }
    return report


def _solve_bias(logits, parent, target_mean):
    lower, upper = -20.0, 20.0
    for _ in range(48):
        middle = (lower + upper) / 2.0
        predicted = (parent * torch.sigmoid(logits + middle)).mean()
        if float(predicted) < target_mean:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def calibration_biases(model, mean, scale, validation, device):
    logits = {task: [] for task in TASK_LABELS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation), 250_000):
            features = validation.features[start:start + 250_000].to(device)
            output = model((features - mean) / scale)
            for task in TASK_LABELS:
                logits[task].append(output[task])
    logits = {task: torch.cat(parts) for task, parts in logits.items()}
    task_targets = targets(validation.labels.to(device))
    biases = {}
    one = torch.ones(len(validation), device=device)
    biases["anchor_click"] = _solve_bias(
        logits["anchor_click"], one, float(task_targets["anchor_click"].mean())
    )
    anchor = torch.sigmoid(logits["anchor_click"] + biases["anchor_click"])
    biases["poi_detail"] = _solve_bias(
        logits["poi_detail"], anchor, float(task_targets["poi_detail"].mean())
    )
    detail = anchor * torch.sigmoid(logits["poi_detail"] + biases["poi_detail"])
    for task in ("poi_favorite", "conversion"):
        biases[task] = _solve_bias(
            logits[task], detail, float(task_targets[task].mean())
        )
    for task in ("negative_feedback", "stay_norm"):
        biases[task] = _solve_bias(
            logits[task], one, float(task_targets[task].mean())
        )
    return biases
