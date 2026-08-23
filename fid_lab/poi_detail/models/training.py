"""Request-aware training and artifact replay for detail module families."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as functional

from ...training.common.tensor_ops import gather_candidates
from ..contracts import DETAIL_TASKS
from .architectures import FAMILY_NAMES, build_family


@dataclass(frozen=True)
class DetailRankerBundle:
    name: str
    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    offsets: dict[str, torch.Tensor]
    offline: dict[str, object]

    def score(self, features, semantic, history, module_kind, chunk=20_000):
        values = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(features), chunk):
                current = features[start : start + chunk]
                outputs = self.model(
                    (current - self.mean) / self.scale,
                    semantic[start : start + chunk], history[start : start + chunk],
                    module_kind[start : start + chunk],
                )
                value = (
                    0.20 * torch.sigmoid(outputs["click"] - self.offsets["click"])
                    + 0.30 * torch.sigmoid(
                        outputs["deep_action"] - self.offsets["deep_action"]
                    )
                    + 0.45 * torch.sigmoid(
                        outputs["transaction"] - self.offsets["transaction"]
                    )
                    - 0.20 * torch.sigmoid(
                        outputs["negative"] - self.offsets["negative"]
                    )
                )
                guardrail = 0.12 * current[:, :, 3] - 0.18 * current[:, :, 10]
                values.append(value + guardrail)
        return torch.cat(values)


def _loss(outputs, labels, masks, positive_weight):
    losses = []
    for index, task in enumerate(DETAIL_TASKS):
        mask = masks[:, :, index]
        point = functional.binary_cross_entropy_with_logits(
            outputs[task], labels[:, :, index],
            pos_weight=positive_weight[index], reduction="none",
        )
        point = (point * mask).sum() / mask.sum().clamp_min(1.0)
        positive = (labels[:, :, index] * mask).sum(1) > 0
        listwise = torch.zeros((), device=labels.device)
        if positive.any():
            listwise = -(
                labels[positive, :, index]
                * functional.log_softmax(outputs[task][positive], dim=1)
            ).sum(1).mean()
        losses.append(point + 0.20 * listwise)
    return sum(weight * loss for weight, loss in zip(
        (0.20, 0.30, 0.35, 0.15), losses
    ))


def _offline(model, features, semantic, history, kinds, labels, masks, mean, scale):
    predictions = {task: [] for task in DETAIL_TASKS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 20_000):
            outputs = model(
                (features[start : start + 20_000] - mean) / scale,
                semantic[start : start + 20_000], history[start : start + 20_000],
                kinds[start : start + 20_000],
            )
            for task in DETAIL_TASKS:
                predictions[task].append(torch.sigmoid(outputs[task]).cpu())
    report = {}
    for index, task in enumerate(DETAIL_TASKS):
        mask = masks[:, :, index].bool().cpu().numpy().reshape(-1)
        target = labels[:, :, index].cpu().numpy().reshape(-1)[mask]
        score = torch.cat(predictions[task]).numpy().reshape(-1)[mask]
        has_both_classes = np.unique(target).size == 2
        report[task] = {
            "auc": float(roc_auc_score(target, score)) if has_both_classes else None,
            "pr_auc": (
                float(average_precision_score(target, score))
                if target.size and target.sum() else None
            ),
            "positive_rate": float(target.mean()),
            "observable_rows": int(target.size),
            "positive_rows": int(target.sum()),
        }
    return report


def train_families(config, world, candidates, features, semantic, response):
    top = response["top_indices"]
    exposed_features = gather_candidates(features, top)
    exposed_semantic = gather_candidates(semantic, top)
    exposed_kinds = gather_candidates(candidates.module_kind, top)
    labels = gather_candidates(response["labels"], top)
    masks = gather_candidates(response["label_masks"], top)
    first, second = int(config.requests * 0.70), int(config.requests * 0.85)
    mean = exposed_features[:first].flatten(0, 1).mean(0)
    scale = exposed_features[:first].flatten(0, 1).std(0).clamp_min(1e-4)
    flat_labels = labels[:first].flatten(0, 1)
    flat_masks = masks[:first].flatten(0, 1)
    positive = (flat_labels * flat_masks).sum(0).clamp_min(1.0)
    negative = (flat_masks.sum(0) - positive).clamp_min(1.0)
    positive_weight = (negative / positive).clamp(max=30.0)
    bundles = {}
    for offset, name in enumerate(FAMILY_NAMES):
        torch.manual_seed(config.seed + 900 + offset)
        model = build_family(name, features.shape[2], config.semantic_dim).to(
            features.device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        generator = torch.Generator(device=features.device).manual_seed(
            config.seed + 910 + offset
        )
        losses = []
        for _ in range(config.train_epochs):
            order = torch.randperm(first, generator=generator, device=features.device)
            for start in range(0, first, config.train_batch_requests):
                request = order[start : start + config.train_batch_requests]
                outputs = model(
                    (exposed_features[request] - mean) / scale,
                    exposed_semantic[request], world.requests.history_sequence[request],
                    exposed_kinds[request],
                )
                loss = _loss(outputs, labels[request], masks[request], positive_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        offsets = {
            task: positive_weight[index].log()
            for index, task in enumerate(DETAIL_TASKS)
        }
        bundles[name] = DetailRankerBundle(
            name, model, mean, scale, offsets,
            {
                "parameters": sum(value.numel() for value in model.parameters()),
                "final_loss": float(np.mean(losses[-max(
                    first // config.train_batch_requests, 1
                ):])),
                "metrics": _offline(
                    model, exposed_features[first:second],
                    exposed_semantic[first:second],
                    world.requests.history_sequence[first:second],
                    exposed_kinds[first:second], labels[first:second],
                    masks[first:second], mean, scale,
                ),
            },
        )
    return bundles


def save_bundle(bundle, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "poi-detail-ranker-v1", "name": bundle.name,
        "feature_width": len(bundle.mean), "semantic_dim": config.semantic_dim,
        "state_dict": bundle.model.state_dict(), "mean": bundle.mean,
        "scale": bundle.scale, "offsets": bundle.offsets,
        "offline": bundle.offline,
    }, path)
    return {"artifact_file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_bundle(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "poi-detail-ranker-v1":
        raise ValueError("unsupported POI Detail artifact")
    model = build_family(
        payload["name"], payload["feature_width"], payload["semantic_dim"]
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return DetailRankerBundle(
        payload["name"], model, payload["mean"].to(device),
        payload["scale"].to(device),
        {name: value.to(device) for name, value in payload["offsets"].items()},
        payload["offline"],
    )
