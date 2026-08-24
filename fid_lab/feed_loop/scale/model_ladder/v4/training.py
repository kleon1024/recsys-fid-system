"""GPU training loop for request-aware single and multi-task Feed rankers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .contracts import LONG_VIEW_TASK, TASKS
from .data import RequestSplit


def _selected_logits(logits, exposed_index):
    rows = torch.arange(len(logits), device=logits.device)
    return logits[rows, exposed_index]


def _targets(labels, masks, task_count):
    if task_count == 1:
        task = TASKS[LONG_VIEW_TASK]
        return labels[:, task.label_index : task.label_index + 1], (
            masks[:, task.label_index : task.label_index + 1]
        )
    values = []
    valid = []
    for task in TASKS:
        values.append(labels[:, task.label_index] / task.scale)
        valid.append(masks[:, task.label_index])
    return torch.stack(values, dim=1), torch.stack(valid, dim=1)


def request_loss(model, batch, listwise_weight=0.10):
    logits = model(batch["candidates"], batch["sequence"])
    selected = _selected_logits(logits, batch["exposed_index"])
    targets, masks = _targets(batch["labels"], batch["masks"], model.task_count)
    losses = []
    tasks = (TASKS[LONG_VIEW_TASK],) if model.task_count == 1 else TASKS
    for index, task in enumerate(tasks):
        if task.kind == "binary":
            values = nn.functional.binary_cross_entropy_with_logits(
                selected[:, index], targets[:, index], reduction="none"
            )
        else:
            values = nn.functional.smooth_l1_loss(
                selected[:, index], targets[:, index], reduction="none"
            )
        valid = masks[:, index].float()
        weighted = values * batch["weights"] * valid
        losses.append(task.loss_weight * weighted.sum() / valid.sum().clamp_min(1.0))
    loss = sum(losses)
    if model.task_count == 1:
        reward = targets[:, 0] * masks[:, 0].float()
        policy_score = logits[:, :, 0]
    else:
        reward = torch.zeros_like(targets[:, 0])
        policy_score = torch.zeros_like(logits[:, :, 0])
        for index, task in enumerate(TASKS):
            reward += task.audit_weight * targets[:, index] * masks[:, index].float()
            prediction = (
                torch.sigmoid(logits[:, :, index])
                if task.kind == "binary" else logits[:, :, index].clamp(0.0, 1.0)
            )
            policy_score += task.audit_weight * prediction
    if reward.abs().sum() > 0:
        chosen_log_probability = _selected_logits(
            torch.log_softmax(policy_score / 0.20, dim=1)[:, :, None],
            batch["exposed_index"],
        ).squeeze(1)
        policy_loss = -(
            chosen_log_probability * reward * batch["weights"]
        ).sum() / reward.abs().sum().clamp_min(1.0)
        loss = loss + listwise_weight * policy_loss
    return loss


@torch.inference_mode()
def validation_loss(model, split, device, batch_size):
    model.eval()
    values = []
    for start in range(0, len(split), batch_size):
        index = torch.arange(start, min(start + batch_size, len(split)))
        values.append(float(request_loss(model, split.batch(index, device))))
    return float(np.mean(values))


def fit_request_model(
    model: nn.Module,
    train: RequestSplit,
    validation: RequestSplit,
    device: torch.device,
    epochs: int,
    seed: int,
    batch_size: int = 2_048,
):
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        losses = []
        for start in range(0, len(train), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = request_loss(model, train.batch(index, device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        current = validation_loss(model, validation, device, batch_size)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": current,
        })
        if current < best_loss:
            best_loss = current
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
