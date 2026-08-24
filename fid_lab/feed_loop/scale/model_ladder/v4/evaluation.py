"""Offline and request-ranking evaluation for V4 Feed models."""

from __future__ import annotations

import numpy as np
import torch

from .....evolution.evaluation.metrics import binary_metrics, grouped_auc
from .contracts import LONG_VIEW_TASK, TASKS


def _supported_binary_metrics(target, score):
    positives = int(np.asarray(target).sum())
    rows = len(target)
    support = {"rows": rows, "positives": positives, "negatives": rows - positives}
    if positives == 0 or positives == rows:
        return {**support, "auc": None, "pr_auc": None, "log_loss": None,
                "ece": None, "ndcg": None, "identifiable": False}
    return {**support, **binary_metrics(target, score), "identifiable": True}


@torch.inference_mode()
def predict_request_model(model, split, device, rows=None, batch_size=512):
    model.eval()
    count = len(split) if rows is None else min(rows, len(split))
    output = []
    for start in range(0, count, batch_size):
        index = torch.arange(start, min(start + batch_size, count))
        batch = split.batch(index, device)
        output.append(model(batch["candidates"], batch["sequence"]).float().cpu())
    return torch.cat(output).numpy()


def _probabilities(logits, task_count):
    output = np.empty_like(logits, dtype=np.float32)
    tasks = (TASKS[LONG_VIEW_TASK],) if task_count == 1 else TASKS
    for index, task in enumerate(tasks):
        if task.kind == "binary":
            output[..., index] = 1.0 / (1.0 + np.exp(-logits[..., index]))
        else:
            output[..., index] = np.clip(logits[..., index], 0.0, 1.0)
    return output


def _rank_scores(probabilities, task_count):
    if task_count == 1:
        return probabilities[..., 0]
    score = np.zeros(probabilities.shape[:2], dtype=np.float32)
    for index, task in enumerate(TASKS):
        score += task.audit_weight * probabilities[..., index]
    return score


def evaluate_request_model(model, split, device, ranking_rows=20_000):
    logits = predict_request_model(model, split, device)
    probability = _probabilities(logits, model.task_count)
    exposed = split.tensors["exposed_index"][split.indices].long().numpy()
    rows = np.arange(len(split))
    selected = probability[rows, exposed]
    labels = split.labels()
    users = split.user_ids()
    long_index = 0 if model.task_count == 1 else LONG_VIEW_TASK
    long_task = TASKS[LONG_VIEW_TASK]
    long_labels = labels[:, long_task.label_index]
    offline = _supported_binary_metrics(long_labels, selected[:, long_index])
    offline["user_gauc"] = grouped_auc(
        long_labels, selected[:, long_index], users
    )
    head_metrics = {}
    if model.task_count > 1:
        for index, task in enumerate(TASKS):
            target = labels[:, task.label_index] / task.scale
            if task.kind == "binary":
                head_metrics[task.name] = _supported_binary_metrics(
                    target, selected[:, index]
                )
            else:
                error = selected[:, index] - target
                head_metrics[task.name] = {
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                }
    count = min(ranking_rows, len(split))
    score = _rank_scores(probability[:count], model.task_count)
    utility = split.tensors["candidate_audit_utility"][
        split.indices[:count]
    ].float().numpy()
    choice = score.argmax(axis=1)
    request_rows = np.arange(count)
    ranking = {
        "requests": count,
        "audit_regret": float(
            (utility.max(axis=1) - utility[request_rows, choice]).mean()
        ),
        "audit_top1_rate": float((choice == utility.argmax(axis=1)).mean()),
    }
    return {"offline": offline, "heads": head_metrics, "ranking": ranking}


@torch.inference_mode()
def sequence_ablation(model, split, device, rows=20_000, seed=20260823):
    count = min(rows, len(split))
    index = torch.arange(count)
    batch = split.batch(index, device)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    baseline = model(batch["candidates"], batch["sequence"])
    shuffled = model(batch["candidates"], batch["sequence"][permutation])
    delta = (torch.sigmoid(baseline) - torch.sigmoid(shuffled)).abs()
    return {
        "rows": count,
        "mean_probability_delta": float(delta.mean()),
        "p95_probability_delta": float(delta.quantile(0.95)),
    }
