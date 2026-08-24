"""PyTorch multi-task ranker and chronological micro-batch trainer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ....evolution.evaluation.metrics import binary_metrics, grouped_auc
from ..contracts import Surface
from ..exchange import TASKS
from ..serving.surfaces import CANDIDATE_FEATURES
from .contracts import FineRankExampleBatch
from .networks import build_network


RANK_VALUE_WEIGHTS = (
    0.05, 0.10, 0.40, 0.45, 0.35, 0.25, 0.45, 0.50,
    0.20, 0.30, 0.35, 0.45, 0.65, 0.80, -0.55, 0.35, 0.60,
)


class MultiTaskRanker(nn.Module):
    def __init__(self, architecture: str = "lr", hidden: int = 64):
        super().__init__()
        inputs = len(CANDIDATE_FEATURES) + len(Surface)
        self.network = build_network(
            architecture, inputs, TASKS, hidden
        )
        self.architecture = architecture
        self.hidden = hidden

    def forward(
        self, features: torch.Tensor, surface: torch.Tensor,
    ) -> torch.Tensor:
        surface_one_hot = torch.nn.functional.one_hot(
            surface.long(), len(Surface)
        ).to(features.dtype)
        return self.network(torch.cat((features, surface_one_hot), dim=-1))


@dataclass
class RankerArtifact:
    model_id: str
    architecture: str
    model: MultiTaskRanker
    training_report: dict[str, object]
    serving_task_weights: tuple[float, ...]

    @torch.inference_mode()
    def score(
        self, features: torch.Tensor, surface: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = features.shape[:2]
        parameter = next(self.model.parameters())
        flat_features = features.reshape(
            -1, features.shape[-1]
        ).to(parameter.device, parameter.dtype)
        flat_surface = surface[:, None].expand(original_shape).reshape(-1).to(
            parameter.device
        )
        logits = self.model(flat_features, flat_surface)
        weights = torch.tensor(
            self.serving_task_weights,
            device=logits.device,
            dtype=logits.dtype,
        )
        value = torch.sigmoid(logits) @ weights
        return (value / weights.abs().sum()).reshape(original_shape)

    def manifest(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "tasks": list(TASKS),
            "features": list(CANDIDATE_FEATURES),
            "training": self.training_report,
            "serving_task_weights": list(self.serving_task_weights),
        }

    def checkpoint(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "hidden": self.model.hidden,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "training_report": self.training_report,
            "serving_task_weights": self.serving_task_weights,
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: dict[str, object], device: str | torch.device,
    ) -> RankerArtifact:
        model = MultiTaskRanker(
            str(checkpoint["architecture"]), int(checkpoint["hidden"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(
            model_id=str(checkpoint["model_id"]),
            architecture=str(checkpoint["architecture"]),
            model=model,
            training_report=dict(checkpoint["training_report"]),
            serving_task_weights=tuple(checkpoint["serving_task_weights"]),
        )


def _task_weights(batch: FineRankExampleBatch, device, selection):
    masks = batch.label_mask[selection].to(device)
    labels = batch.labels[selection].to(device)
    mature_count = masks.sum(dim=(0, 1))
    positive_count = (labels * masks).sum(dim=(0, 1))
    negative_count = mature_count - positive_count
    reliable = (
        (mature_count >= 100)
        & (positive_count >= 5)
        & (negative_count >= 5)
    )
    values = torch.tensor(RANK_VALUE_WEIGHTS, device=device)
    return values * reliable, mature_count, positive_count, reliable


@torch.inference_mode()
def _offline_metrics(model, batch, start, device):
    if start >= len(batch.request_id):
        return {"requests": 0, "tasks": {}}
    features = batch.features[start:].to(device, dtype=torch.float32)
    surface = batch.surface[start:].to(device)
    requests, items = features.shape[:2]
    flat_surface = surface[:, None].expand(requests, items).reshape(-1)
    probability = torch.sigmoid(model(
        features.reshape(-1, features.shape[-1]), flat_surface
    )).reshape(requests, items, len(TASKS)).cpu().numpy()
    labels = batch.labels[start:].cpu().numpy()
    masks = batch.label_mask[start:].cpu().numpy()
    valid = (batch.item_ids[start:] >= 0).cpu().numpy()
    users = np.repeat(
        batch.user_id[start:].cpu().numpy(), items
    ).reshape(requests, items)
    report = {}
    for index, task in enumerate(TASKS):
        selected = masks[:, :, index] & valid
        target = labels[:, :, index][selected]
        score = probability[:, :, index][selected]
        group = users[selected]
        if len(target) == 0 or np.unique(target).size < 2:
            report[task] = {
                "rows": int(len(target)),
                "positive": int(target.sum()) if len(target) else 0,
                "auc": None,
                "gauc": None,
            }
            continue
        report[task] = {
            "rows": int(len(target)),
            "positive": int(target.sum()),
            **binary_metrics(target, score),
            "user_gauc": grouped_auc(target, score, group),
        }
    return {"requests": requests, "tasks": report}


def _request_loss(
    model, features, surface, labels, masks, propensity, selected, valid,
    task_weights,
):
    requests, items = features.shape[:2]
    flat_surface = surface[:, None].expand(requests, items).reshape(-1)
    logits = model(features.reshape(-1, features.shape[-1]), flat_surface)
    logits = logits.reshape(requests, items, len(TASKS))
    loss_values = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    ips = (1.0 / propensity.clamp_min(0.05)).clamp_max(10.0)
    pointwise = (loss_values * masks * ips[:, :, None]).sum()
    pointwise = pointwise / masks.sum().clamp_min(1)
    denominator = task_weights.abs().sum().clamp_min(1e-6)
    value_score = (logits @ task_weights) / denominator
    observed_value = (labels * masks * task_weights).sum(dim=2)
    positive = (observed_value * selected).sum(dim=1) > 0
    selected_score = (value_score * selected).sum(dim=1)
    comparison = valid & ~selected & positive[:, None]
    pair_values = torch.nn.functional.softplus(
        -(selected_score[:, None] - value_score)
    )
    pairwise = (
        pair_values[comparison].mean()
        if comparison.any() else pointwise.new_zeros(())
    )
    selected_position = selected.float().argmax(dim=1)
    listwise = (
        torch.nn.functional.cross_entropy(
            value_score[positive].masked_fill(~valid[positive], -1e9),
            selected_position[positive],
        )
        if positive.any() else pointwise.new_zeros(())
    )
    return pointwise + 0.15 * pairwise + 0.05 * listwise, (
        pointwise, pairwise, listwise
    )


def train_fine_ranker(
    batch: FineRankExampleBatch,
    *,
    model_id: str,
    architecture: str = "lr",
    epochs: int = 3,
    microbatch_rows: int = 4_096,
    learning_rate: float = 3e-3,
    device: str | torch.device | None = None,
    seed: int = 20260824,
) -> RankerArtifact:
    target_device = torch.device(device or batch.features.device)
    features = batch.features.to(target_device, dtype=torch.float32)
    surface = batch.surface.to(target_device)
    labels = batch.labels.to(target_device)
    masks = batch.label_mask.to(target_device)
    propensity = batch.examination_propensity.to(target_device)
    propensity = propensity * batch.request_sampling_probability.to(
        target_device
    )[:, None]
    selected = batch.selected.to(target_device)
    valid = batch.item_ids.to(target_device) >= 0
    if not masks.any():
        raise ValueError("no mature labels are available for ranker training")
    if not torch.all(batch.step[1:] >= batch.step[:-1]):
        raise ValueError("fine-rank requests must be chronological")
    unique_steps = torch.unique_consecutive(batch.step)
    if len(unique_steps) > 1:
        test_step_index = min(
            max(int(len(unique_steps) * 0.80), 1),
            len(unique_steps) - 1,
        )
        test_step = unique_steps[test_step_index]
        train_requests = int(torch.searchsorted(
            batch.step, test_step, right=False
        ))
    else:
        train_requests = max(int(len(features) * 0.80), 1)
        if len(features) > 1:
            train_requests = min(train_requests, len(features) - 1)
    training_selection = slice(0, train_requests)
    task_weights, mature_count, positive_count, reliable = _task_weights(
        batch, target_device, training_selection
    )
    serving_weights = tuple(float(value) for value in task_weights)
    if not any(serving_weights):
        raise ValueError("no task has enough mature positive and negative labels")
    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MultiTaskRanker(architecture).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses, components = [], []
    request_batch = max(microbatch_rows // batch.item_ids.shape[1], 1)
    model.train()
    for _epoch in range(epochs):
        for start in range(0, train_requests, request_batch):
            selection = slice(
                start, min(start + request_batch, train_requests)
            )
            loss, parts = _request_loss(
                model, features[selection], surface[selection],
                labels[selection], masks[selection], propensity[selection],
                selected[selection], valid[selection], task_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append(tuple(float(value.detach()) for value in parts))
    model.eval()
    offline = _offline_metrics(
        model, batch, train_requests, target_device
    )
    window = max(len(losses) // max(epochs, 1), 1)
    report = {
        "source_schema": batch.manifest.schema_version,
        "source_policy": batch.manifest.served_policy,
        "rows": int(valid.sum()),
        "mature_labels": int(masks.sum()),
        "epochs": epochs,
        "microbatch_rows": microbatch_rows,
        "loss_first": sum(losses[:window]) / window,
        "loss_last": sum(losses[-window:]) / window,
        "optimizer": "torch.optim.AdamW",
        "training_device": str(target_device),
        "training_seed": seed,
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "chronological_split": {
            "train_requests": train_requests,
            "test_requests": len(features) - train_requests,
            "train_max_step": int(batch.step[:train_requests].max()),
            "test_min_step": int(batch.step[train_requests:].min()),
            "mode": (
                "whole_step" if len(unique_steps) > 1
                else "within_single_step_fallback"
            ),
        },
        "offline": offline,
        "objective": "clipped_ips_bce + 0.15_pairwise + 0.05_listwise",
        "loss_components_last": {
            "pointwise": components[-1][0],
            "pairwise": components[-1][1],
            "listwise": components[-1][2],
        },
    }
    report["task_maturity"] = {
        task: {
            "mature": int(mature_count[index]),
            "positive": int(positive_count[index]),
            "used_for_serving": bool(reliable[index]),
        }
        for index, task in enumerate(TASKS)
    }
    return RankerArtifact(
        model_id, architecture, model, report, serving_weights
    )
