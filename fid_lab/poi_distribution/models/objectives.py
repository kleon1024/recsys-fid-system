"""Multi-task objectives for the POI distribution rankers."""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import TASK_LABELS


BINARY_TASKS = tuple(task for task in TASK_LABELS if task != "stay_norm")
TASK_WEIGHTS = {
    "anchor_click": 0.25,
    "poi_detail": 0.25,
    "poi_favorite": 0.10,
    "conversion": 0.20,
    "negative_feedback": 0.10,
    "stay_norm": 0.10,
}


def task_probabilities(outputs, calibration_biases=None):
    calibration_biases = calibration_biases or {}
    raw = {
        task: torch.sigmoid(outputs[task] + calibration_biases.get(task, 0.0))
        for task in BINARY_TASKS
    }
    anchor = raw["anchor_click"]
    detail = anchor * raw["poi_detail"]
    return {
        "anchor_click": anchor,
        "poi_detail": detail,
        "poi_favorite": detail * raw["poi_favorite"],
        "conversion": detail * raw["conversion"],
        "negative_feedback": raw["negative_feedback"],
        "stay_norm": torch.sigmoid(
            outputs["stay_norm"] + calibration_biases.get("stay_norm", 0.0)
        ),
    }


def targets(labels):
    return {
        task: labels[:, index] / 180.0 if task == "stay_norm" else labels[:, index]
        for task, index in TASK_LABELS.items()
    }


def positive_weights(labels):
    values = targets(labels)
    return {
        task: ((1.0 - target.mean()) / target.mean().clamp_min(1e-6)).clamp(1, 50)
        for task, target in values.items()
        if task != "stay_norm"
    }


def multi_task_loss(outputs, labels, ips_weights, task_positive_weights):
    task_targets = targets(labels)
    probabilities = task_probabilities(outputs)
    total = torch.zeros((), device=labels.device)
    for task in TASK_LABELS:
        target = task_targets[task]
        if task == "stay_norm":
            point = nn.functional.smooth_l1_loss(
                probabilities[task], target, reduction="none"
            )
        else:
            probability = probabilities[task].clamp(1e-6, 1 - 1e-6)
            point = -(
                task_positive_weights[task] * target * probability.log()
                + (1.0 - target) * (1.0 - probability).log()
            )
        total += TASK_WEIGHTS[task] * (point * ips_weights).mean()
    return total
