"""Probability-space and loss contracts for the posting behavior funnel."""

from __future__ import annotations

import torch
from torch.nn import functional as functional

from .contracts import FEED_POSTING_TASKS


MASKED_CONDITIONAL = "masked_conditional_v1"
ENTIRE_SPACE_CASCADE = "entire_space_cascade_v1"
CASCADE_TASKS = ("click", "create", "publish")


def entire_space_targets(labels, masks):
    """Lift joint create/publish outcomes into the valid impression space."""
    targets = labels.clone()
    observed = masks.clone()
    exposed = masks[:, :, 0]
    observed[:, :, 1] = exposed
    observed[:, :, 2] = exposed
    return targets, observed


def task_probabilities(outputs, objective, offsets=None):
    offsets = offsets or {}
    conditional = {
        task: torch.sigmoid(outputs[task] - offsets.get(task, 0.0))
        for task in FEED_POSTING_TASKS
    }
    if objective == MASKED_CONDITIONAL:
        return conditional
    if objective != ENTIRE_SPACE_CASCADE:
        raise ValueError(f"unknown Feed Posting objective: {objective}")
    return {
        **conditional,
        "create": conditional["click"] * conditional["create"],
        "publish": (
            conditional["click"]
            * conditional["create"]
            * conditional["publish"]
        ),
    }


def _masked_conditional_loss(outputs, labels, masks, positive_weight):
    losses = []
    for index, task in enumerate(FEED_POSTING_TASKS):
        point = functional.binary_cross_entropy_with_logits(
            outputs[task], labels[:, :, index],
            pos_weight=positive_weight[index], reduction="none",
        )
        point = (point * masks[:, :, index]).sum() / (
            masks[:, :, index].sum().clamp_min(1)
        )
        positive = labels[:, :, index].sum(1) > 0
        listwise = torch.zeros((), device=labels.device)
        if positive.any():
            observed_logits = outputs[task][positive].masked_fill(
                masks[positive, :, index] == 0, -1e9
            )
            listwise = -(
                labels[positive, :, index]
                * functional.log_softmax(observed_logits, dim=1)
            ).sum(1).mean()
        losses.append(point + 0.25 * listwise)
    return losses


def _entire_space_loss(outputs, labels, masks, positive_weight):
    targets, observed = entire_space_targets(labels, masks)
    probabilities = task_probabilities(outputs, ENTIRE_SPACE_CASCADE)
    losses = []
    for index, task in enumerate(FEED_POSTING_TASKS):
        probability = probabilities[task].clamp(1e-6, 1.0 - 1e-6)
        target = targets[:, :, index]
        point = -(
            positive_weight[index] * target * probability.log()
            + (1.0 - target) * (-probability).log1p()
        )
        point = (point * observed[:, :, index]).sum() / (
            observed[:, :, index].sum().clamp_min(1)
        )
        positive = target.sum(1) > 0
        listwise = torch.zeros((), device=labels.device)
        if positive.any():
            log_probability = probability[positive].log().masked_fill(
                observed[positive, :, index] == 0, -1e9
            )
            listwise = -(
                target[positive]
                * functional.log_softmax(log_probability, dim=1)
            ).sum(1).mean()
        losses.append(point + 0.25 * listwise)
    return losses


def multitask_loss(outputs, labels, masks, positive_weight, objective):
    losses = (
        _entire_space_loss(outputs, labels, masks, positive_weight)
        if objective == ENTIRE_SPACE_CASCADE
        else _masked_conditional_loss(outputs, labels, masks, positive_weight)
    )
    weights = (0.22, 0.18, 0.25, 0.20, 0.15)
    return sum(
        weight * loss for weight, loss in zip(weights, losses, strict=True)
    )
