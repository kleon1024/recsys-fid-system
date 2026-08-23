"""Linear, W&D, DIN, and Transformer+MMoE Feed-posting rankers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as functional

from ..multitask import MultiGateMixtureOfExperts
from ..training.common.request_rankers import RequestLinearRanker
from ..training.common.tensor_ops import gather_candidates
from .contracts import FEED_POSTING_TASKS, FeedPostingConfig


class LinearRanker(RequestLinearRanker):
    def __init__(self, width, semantic_dim):
        del semantic_dim
        super().__init__(width, FEED_POSTING_TASKS)


class WideDeepRanker(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        del semantic_dim
        self.wide = nn.ModuleDict({
            task: nn.Linear(width, 1) for task in FEED_POSTING_TASKS
        })
        self.deep = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(32, 1) for task in FEED_POSTING_TASKS
        })

    def forward(self, features, candidate_semantic, history):
        del candidate_semantic, history
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        deep = self.deep(flat)
        return {
            task: (
                self.wide[task](flat) + self.heads[task](deep)
            ).reshape(shape)
            for task in FEED_POSTING_TASKS
        }


class DINRanker(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        self.query = nn.Linear(semantic_dim, semantic_dim)
        self.key = nn.Linear(semantic_dim, semantic_dim)
        self.shared = nn.Sequential(
            nn.Linear(width + 2 * semantic_dim, 64),
            nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(32, 1) for task in FEED_POSTING_TASKS
        })
        self.scale = sqrt(semantic_dim)

    def forward(self, features, candidate_semantic, history):
        query = self.query(candidate_semantic)
        key = self.key(history)
        attention = torch.softmax(
            torch.einsum("bkd,bld->bkl", query, key) / self.scale, dim=2
        )
        pooled = torch.einsum("bkl,bld->bkd", attention, history)
        state = self.shared(torch.cat(
            (features, candidate_semantic, pooled), dim=2
        ).flatten(0, 1))
        shape = features.shape[:2]
        return {task: head(state).reshape(shape) for task, head in self.heads.items()}


class TransformerMMoERanker(nn.Module):
    def __init__(self, width, semantic_dim):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            semantic_dim, 4, semantic_dim * 2, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.query = nn.Linear(semantic_dim, semantic_dim)
        self.mmoe = MultiGateMixtureOfExperts(
            width + 2 * semantic_dim, FEED_POSTING_TASKS, 4, 32
        )
        self.scale = sqrt(semantic_dim)

    def forward(self, features, candidate_semantic, history):
        encoded = self.encoder(history)
        query = self.query(candidate_semantic)
        attention = torch.softmax(
            torch.einsum("bkd,bld->bkl", query, encoded) / self.scale, dim=2
        )
        pooled = torch.einsum("bkl,bld->bkd", attention, encoded)
        inputs = torch.cat(
            (features, candidate_semantic, pooled), dim=2
        ).flatten(0, 1)
        outputs = self.mmoe(inputs)
        shape = features.shape[:2]
        return {
            name: value.reshape(*shape, -1) if name.startswith("gate:")
            else value.reshape(shape)
            for name, value in outputs.items()
        }


MODEL_FACTORIES = {
    "linear": LinearRanker,
    "wide_deep": WideDeepRanker,
    "din": DINRanker,
    "transformer_mmoe": TransformerMMoERanker,
}


@dataclass(frozen=True)
class FeedPostingBundle:
    name: str
    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    logit_offsets: dict[str, torch.Tensor]
    offline: dict[str, object]

    def score(self, features, candidate_semantic, history, chunk=20_000):
        values = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(features), chunk):
                normalized = (features[start : start + chunk] - self.mean) / self.scale
                outputs = self.model(
                    normalized,
                    candidate_semantic[start : start + chunk],
                    history[start : start + chunk],
                )
                task_value = (
                    0.20 * torch.sigmoid(
                        outputs["click"] - self.logit_offsets["click"]
                    )
                    + 0.25 * torch.sigmoid(
                        outputs["create"] - self.logit_offsets["create"]
                    )
                    + 0.40 * torch.sigmoid(
                        outputs["publish"] - self.logit_offsets["publish"]
                    )
                    + 0.15 * torch.sigmoid(
                        outputs["quality"] - self.logit_offsets["quality"]
                    )
                )
                observable_guardrail = (
                    0.10 * features[start : start + chunk, :, 5]
                    - 0.12 * features[start : start + chunk, :, 7]
                )
                values.append(task_value + observable_guardrail)
        return torch.cat(values)


def _loss(outputs, labels, positive_weight):
    losses = []
    for index, task in enumerate(FEED_POSTING_TASKS):
        point = functional.binary_cross_entropy_with_logits(
            outputs[task], labels[:, :, index],
            pos_weight=positive_weight[index],
        )
        positive = labels[:, :, index].sum(1) > 0
        listwise = torch.zeros((), device=labels.device)
        if positive.any():
            listwise = -(
                labels[positive, :, index]
                * functional.log_softmax(outputs[task][positive], dim=1)
            ).sum(1).mean()
        losses.append(point + 0.25 * listwise)
    weights = (0.30, 0.25, 0.30, 0.15)
    return sum(weight * loss for weight, loss in zip(weights, losses))


def _offline(model, features, semantic, history, labels, mean, scale):
    predictions = {task: [] for task in FEED_POSTING_TASKS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 20_000):
            outputs = model(
                (features[start : start + 20_000] - mean) / scale,
                semantic[start : start + 20_000], history[start : start + 20_000],
            )
            for task in FEED_POSTING_TASKS:
                predictions[task].append(torch.sigmoid(outputs[task]).flatten().cpu())
    target = labels.flatten(0, 1).cpu().numpy()
    report = {}
    for index, task in enumerate(FEED_POSTING_TASKS):
        score = torch.cat(predictions[task]).numpy()
        report[task] = {
            "auc": float(roc_auc_score(target[:, index], score)),
            "pr_auc": float(average_precision_score(target[:, index], score)),
            "positive_rate": float(target[:, index].mean()),
        }
    return report


def train_models(
    config, features, semantic, history, top, labels,
    model_names=tuple(MODEL_FACTORIES),
):
    exposed_features = gather_candidates(features, top)
    exposed_semantic = gather_candidates(semantic, top)
    exposed_labels = gather_candidates(labels, top)
    first, second = int(config.requests * 0.70), int(config.requests * 0.85)
    mean = exposed_features[:first].flatten(0, 1).mean(0)
    scale = exposed_features[:first].flatten(0, 1).std(0).clamp_min(1e-4)
    flat_labels = exposed_labels[:first].flatten(0, 1)
    positives = flat_labels.sum(0).clamp_min(1.0)
    positive_weight = ((len(flat_labels) - positives) / positives).clamp(max=30.0)
    bundles = {}
    for offset, (name, factory) in enumerate(MODEL_FACTORIES.items()):
        if name not in model_names:
            continue
        torch.manual_seed(config.seed + 400 + offset)
        model = factory(features.shape[2], config.semantic_dim).to(features.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        generator = torch.Generator(device=features.device).manual_seed(
            config.seed + 500 + offset
        )
        losses = []
        for _ in range(config.train_epochs):
            order = torch.randperm(first, generator=generator, device=features.device)
            for start in range(0, first, config.train_batch_requests):
                request = order[start : start + config.train_batch_requests]
                outputs = model(
                    (exposed_features[request] - mean) / scale,
                    exposed_semantic[request], history[request],
                )
                loss = _loss(outputs, exposed_labels[request], positive_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        offline = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "final_loss": float(np.mean(losses[-max(first // config.train_batch_requests, 1):])),
            "metrics": _offline(
                model, exposed_features[first:second], exposed_semantic[first:second],
                history[first:second], exposed_labels[first:second], mean, scale,
            ),
        }
        offsets = {
            task: positive_weight[index].log()
            for index, task in enumerate(FEED_POSTING_TASKS)
        }
        offline["weighted_loss_logit_offsets"] = {
            task: float(offset) for task, offset in offsets.items()
        }
        bundles[name] = FeedPostingBundle(
            name, model, mean, scale, offsets, offline
        )
    return bundles


def save_bundle(bundle, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "feed-posting-model-v1", "name": bundle.name,
        "feature_width": len(bundle.mean), "semantic_dim": config.semantic_dim,
        "state_dict": bundle.model.state_dict(), "mean": bundle.mean,
        "scale": bundle.scale, "logit_offsets": bundle.logit_offsets,
        "offline": bundle.offline,
    }, path)
    return {"artifact_file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_bundle(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "feed-posting-model-v1":
        raise ValueError("unsupported Feed-posting artifact")
    model = MODEL_FACTORIES[payload["name"]](
        payload["feature_width"], payload["semantic_dim"]
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return FeedPostingBundle(
        payload["name"], model, payload["mean"].to(device),
        payload["scale"].to(device),
        {name: value.to(device) for name, value in payload["logit_offsets"].items()},
        payload["offline"],
    )
