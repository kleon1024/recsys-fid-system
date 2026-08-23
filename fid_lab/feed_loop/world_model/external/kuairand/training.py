"""GPU training and time-held-out evaluation for external behavior kernels."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
import torch
from torch import nn

from .contracts import FEEDBACK_NAMES


TASK_WEIGHTS = torch.tensor((0.45, 1.0, 0.25, 0.10, 0.10, 0.10, 0.15, 1.0))


def _batch(split, indices, device, include_history=True):
    result = {
        "sparse": split.sparse[indices].to(device, non_blocking=True),
        "dense": split.dense[indices].to(device, non_blocking=True),
        "labels": split.labels[indices].to(device, non_blocking=True),
    }
    if include_history:
        result.update({
            "history_items": split.history_items[indices].to(
                device, non_blocking=True
            ),
            "history_feedback": split.history_feedback[indices].to(
                device, non_blocking=True
            ),
        })
    return result


def _loss(logits, labels, positive_weight, task_weights):
    binary = nn.functional.binary_cross_entropy_with_logits(
        logits[:, :7], labels[:, :7], pos_weight=positive_weight,
        reduction="none",
    ).mean(dim=0)
    stay = nn.functional.smooth_l1_loss(
        torch.sigmoid(logits[:, 7]), labels[:, 7]
    )
    return (binary * task_weights[:7]).sum() + task_weights[7] * stay


@torch.inference_mode()
def validation_loss(model, split, device, positive_weight, batch_size=4_096):
    model.eval()
    losses = []
    weights = TASK_WEIGHTS.to(device)
    for start in range(0, len(split), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split)))
        batch = _batch(split, indices, device)
        logits = model(
            batch["sparse"], batch["dense"], batch["history_items"],
            batch["history_feedback"],
        )
        losses.append(float(_loss(logits, batch["labels"], positive_weight, weights)))
    return float(np.mean(losses))


def fit_behavior_model(model, train, validation, device, epochs, seed,
                       batch_size=2_048):
    model.to(device)
    positives = train.labels[:, :7].sum(dim=0).clamp_min(1.0)
    positive_weight = ((len(train) - positives) / positives).clamp_max(100.0).to(device)
    task_weights = TASK_WEIGHTS.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best = float("inf")
    best_state = None
    history = []
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), batch_size):
            indices = order[start:start + batch_size]
            batch = _batch(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    batch["sparse"], batch["dense"], batch["history_items"],
                    batch["history_feedback"],
                )
                loss = _loss(logits, batch["labels"], positive_weight, task_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        held_out = validation_loss(model, validation, device, positive_weight)
        history.append({
            "epoch": epoch + 1,
            "training_loss": float(np.mean(losses)),
            "validation_loss": held_out,
        })
        if held_out < best:
            best = held_out
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, perf_counter() - started


@torch.inference_mode()
def predict_behavior(model, split, device, batch_size=4_096,
                     history_permutation=None):
    model.eval()
    output = []
    for start in range(0, len(split), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split)))
        batch = _batch(split, indices, device)
        if history_permutation is not None:
            source = history_permutation[indices]
            batch["history_items"] = split.history_items[source].to(device)
            batch["history_feedback"] = split.history_feedback[source].to(device)
        logits = model(
            batch["sparse"], batch["dense"], batch["history_items"],
            batch["history_feedback"],
        )
        output.append(torch.cat((torch.sigmoid(logits[:, :7]),
                                 torch.sigmoid(logits[:, 7:])), dim=1).cpu())
    return torch.cat(output).numpy()


def behavior_metrics(labels, predictions):
    metrics = {}
    for index, name in enumerate(FEEDBACK_NAMES):
        score = np.clip(predictions[:, index], 1e-7, 1.0 - 1e-7)
        metrics[name] = {
            "auc": float(roc_auc_score(labels[:, index], score)),
            "log_loss": float(log_loss(labels[:, index], score)),
            "positive_rate": float(labels[:, index].mean()),
        }
    metrics["stay_norm"] = {
        "mae": float(mean_absolute_error(labels[:, 7], predictions[:, 7]))
    }
    return metrics
