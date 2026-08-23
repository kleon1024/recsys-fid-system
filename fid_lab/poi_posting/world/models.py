"""Actual linear, Wide & Deep, and MMoE posting rankers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as functional

from ...multitask import MultiGateMixtureOfExperts
from ...training.tensor_ops import gather_candidates
from .contracts import POSTING_TASKS, PostingWorldConfig


class PostingLinear(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.heads = nn.ModuleDict({task: nn.Linear(width, 1) for task in POSTING_TASKS})

    def forward(self, features):
        return {task: head(features).squeeze(1) for task, head in self.heads.items()}


class PostingWideDeep(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wide = nn.ModuleDict({task: nn.Linear(width, 1) for task in POSTING_TASKS})
        self.deep = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.heads = nn.ModuleDict({task: nn.Linear(32, 1) for task in POSTING_TASKS})

    def forward(self, features):
        deep = self.deep(features)
        return {
            task: self.wide[task](features).squeeze(1)
            + self.heads[task](deep).squeeze(1)
            for task in POSTING_TASKS
        }


class PostingMMoE(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = MultiGateMixtureOfExperts(width, POSTING_TASKS, 4, 32)

    def forward(self, features):
        return self.network(features)


MODEL_FACTORIES = {
    "linear": PostingLinear,
    "wide_deep": PostingWideDeep,
    "mmoe": PostingMMoE,
}


@dataclass(frozen=True)
class PostingModelBundle:
    name: str
    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    offline: dict[str, object]

    def score(self, features, chunk=500_000):
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        values = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(flat), chunk):
                normalized = (flat[start : start + chunk] - self.mean) / self.scale
                outputs = self.model(normalized)
                values.append(
                    0.45 * torch.sigmoid(outputs["select"])
                    + 0.35 * torch.sigmoid(outputs["publish"])
                    + 0.20 * torch.sigmoid(outputs["relevance"])
                )
        return torch.cat(values).reshape(shape)


def _request_loss(outputs, labels, positive_weight):
    losses = []
    for index, task in enumerate(POSTING_TASKS):
        logit = outputs[task].reshape(labels.shape[:2])
        point = functional.binary_cross_entropy_with_logits(
            logit,
            labels[:, :, index],
            pos_weight=positive_weight[index],
        )
        positive_request = labels[:, :, index].sum(dim=1) > 0
        if positive_request.any():
            listwise = -(
                labels[positive_request, :, index]
                * functional.log_softmax(logit[positive_request], dim=1)
            ).sum(dim=1).mean()
        else:
            listwise = torch.zeros((), device=labels.device)
        losses.append(point + 0.25 * listwise)
    return 0.45 * losses[0] + 0.35 * losses[1] + 0.20 * losses[2]


def _offline_metrics(model, features, labels, mean, scale):
    model.eval()
    flat = features.flatten(0, 1)
    predictions = {task: [] for task in POSTING_TASKS}
    with torch.inference_mode():
        for start in range(0, len(flat), 500_000):
            outputs = model((flat[start : start + 500_000] - mean) / scale)
            for task in POSTING_TASKS:
                predictions[task].append(torch.sigmoid(outputs[task]).cpu())
    target = labels.flatten(0, 1).cpu().numpy()
    report = {}
    for index, task in enumerate(POSTING_TASKS):
        score = torch.cat(predictions[task]).numpy()
        report[task] = {
            "auc": float(roc_auc_score(target[:, index], score)),
            "pr_auc": float(average_precision_score(target[:, index], score)),
            "positive_rate": float(target[:, index].mean()),
        }
    return report


def train_posting_models(
    config: PostingWorldConfig, features, top_indices, labels
):
    exposed_features = gather_candidates(features, top_indices)
    exposed_labels = gather_candidates(labels, top_indices)
    first = int(config.requests * 0.70)
    second = int(config.requests * 0.85)
    train_features = exposed_features[:first]
    train_labels = exposed_labels[:first]
    validation_features = exposed_features[first:second]
    validation_labels = exposed_labels[first:second]
    mean = train_features.flatten(0, 1).mean(dim=0)
    scale = train_features.flatten(0, 1).std(dim=0).clamp_min(1e-4)
    flat_labels = train_labels.flatten(0, 1)
    positives = flat_labels.sum(dim=0).clamp_min(1.0)
    positive_weight = ((len(flat_labels) - positives) / positives).clamp(max=30.0)
    requests_per_batch = max(
        config.train_batch_pairs // config.exposed_candidates, 1
    )
    bundles = {}
    for offset, (name, factory) in enumerate(MODEL_FACTORIES.items()):
        torch.manual_seed(config.seed + 200 + offset)
        model = factory(features.shape[2]).to(features.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        generator = torch.Generator(device=features.device).manual_seed(
            config.seed + 300 + offset
        )
        losses = []
        for _ in range(config.train_epochs):
            order = torch.randperm(first, generator=generator, device=features.device)
            for start in range(0, first, requests_per_batch):
                request_index = order[start : start + requests_per_batch]
                batch_features = train_features[request_index]
                batch_labels = train_labels[request_index]
                normalized = (batch_features.flatten(0, 1) - mean) / scale
                outputs = model(normalized)
                loss = _request_loss(outputs, batch_labels, positive_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        offline = {
            "train_requests": first,
            "validation_requests": second - first,
            "epochs": config.train_epochs,
            "final_loss": float(np.mean(losses[-max(first // requests_per_batch, 1):])),
            "metrics": _offline_metrics(
                model, validation_features, validation_labels, mean, scale
            ),
        }
        bundles[name] = PostingModelBundle(name, model, mean, scale, offline)
    return bundles


def save_posting_bundle(bundle, path: Path, config: PostingWorldConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "poi-posting-model-v1",
        "name": bundle.name,
        "feature_width": len(bundle.mean),
        "state_dict": bundle.model.state_dict(),
        "mean": bundle.mean,
        "scale": bundle.scale,
        "config": {
            "seed": config.seed,
            "requests": config.requests,
            "train_epochs": config.train_epochs,
        },
        "offline": bundle.offline,
    }
    torch.save(payload, path)
    return {
        "artifact_file": path.name,
        "artifact_sha256": sha256(path.read_bytes()).hexdigest(),
        "schema": payload["schema"],
    }


def load_posting_bundle(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "poi-posting-model-v1":
        raise ValueError("unsupported POI posting artifact")
    model = MODEL_FACTORIES[payload["name"]](payload["feature_width"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return PostingModelBundle(
        payload["name"],
        model,
        payload["mean"].to(device),
        payload["scale"].to(device),
        payload["offline"],
    )
