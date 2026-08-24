"""Linear, W&D, DIN, and Transformer+MMoE Feed-posting rankers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch
from torch import nn

from ..multitask import MultiGateMixtureOfExperts
from ..training.common.request_rankers import RequestLinearRanker
from ..training.common.tensor_ops import gather_candidates
from .contracts import FEED_POSTING_TASKS
from .objectives import (
    ENTIRE_SPACE_CASCADE,
    MASKED_CONDITIONAL,
    entire_space_targets,
    multitask_loss,
    task_probabilities,
)


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
            nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(64, 1) for task in FEED_POSTING_TASKS
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
            nn.Linear(width + 2 * semantic_dim, 128),
            nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(64, 1) for task in FEED_POSTING_TASKS
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
            semantic_dim, 4, semantic_dim * 4, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.query = nn.Linear(semantic_dim, semantic_dim)
        self.mmoe = MultiGateMixtureOfExperts(
            width + 2 * semantic_dim, FEED_POSTING_TASKS, 8, 64
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


def _ece(target, score, bins=20):
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        selected = bucket == index
        if selected.any():
            value += selected.mean() * abs(
                score[selected].mean() - target[selected].mean()
            )
    return float(value)


@dataclass(frozen=True)
class FeedPostingBundle:
    name: str
    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    logit_offsets: dict[str, torch.Tensor]
    offline: dict[str, object]
    objective: str = MASKED_CONDITIONAL

    def probabilities(
        self, features, candidate_semantic, history, chunk=20_000,
    ):
        values = {task: [] for task in FEED_POSTING_TASKS}
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(features), chunk):
                normalized = (
                    features[start : start + chunk] - self.mean
                ) / self.scale
                outputs = self.model(
                    normalized,
                    candidate_semantic[start : start + chunk],
                    history[start : start + chunk],
                )
                probabilities = task_probabilities(
                    outputs, self.objective, self.logit_offsets
                )
                for task in FEED_POSTING_TASKS:
                    values[task].append(probabilities[task])
        return {task: torch.cat(parts) for task, parts in values.items()}

    def score(self, features, candidate_semantic, history, chunk=20_000):
        probabilities = self.probabilities(
            features, candidate_semantic, history, chunk
        )
        task_value = (
            0.20 * probabilities["click"]
            + 0.25 * probabilities["create"]
            + 0.40 * probabilities["publish"]
            + 0.15 * probabilities["quality"]
            - 0.25 * probabilities["risk"]
        )
        observable_guardrail = 0.10 * features[:, :, 5] - 0.12 * features[:, :, 7]
        return task_value + observable_guardrail


def _offline(
    model, features, semantic, history, labels, masks, mean, scale, objective,
):
    predictions = {task: [] for task in FEED_POSTING_TASKS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 20_000):
            outputs = model(
                (features[start : start + 20_000] - mean) / scale,
                semantic[start : start + 20_000], history[start : start + 20_000],
            )
            probabilities = task_probabilities(outputs, objective)
            for task in FEED_POSTING_TASKS:
                predictions[task].append(probabilities[task].flatten().cpu())
    if objective == ENTIRE_SPACE_CASCADE:
        labels, masks = entire_space_targets(labels, masks)
    target = labels.flatten(0, 1).cpu().numpy()
    observed = masks.flatten(0, 1).cpu().numpy()
    report = {}
    for index, task in enumerate(FEED_POSTING_TASKS):
        mask = observed[:, index] > 0
        score = torch.cat(predictions[task]).numpy()[mask]
        task_target = target[mask, index]
        if not mask.any():
            report[task] = {
                "auc": None,
                "pr_auc": None,
                "positive_rate": None,
                "observed_rows": 0,
            }
            continue
        report[task] = {
            "auc": (
                float(roc_auc_score(task_target, score))
                if np.unique(task_target).size == 2 else None
            ),
            "pr_auc": (
                float(average_precision_score(task_target, score))
                if task_target.sum() else None
            ),
            "positive_rate": float(task_target.mean()),
            "observed_rows": int(mask.sum()),
            "log_loss": float(log_loss(task_target, score, labels=[0, 1])),
            "ece_20": _ece(task_target, score),
        }
    return report


def train_models(
    config, features, semantic, history, top, labels, label_masks=None,
    model_names=tuple(MODEL_FACTORIES),
):
    exposed_features = gather_candidates(features, top)
    exposed_semantic = gather_candidates(semantic, top)
    exposed_labels = gather_candidates(labels, top)
    exposed_masks = (
        torch.ones_like(exposed_labels)
        if label_masks is None else gather_candidates(label_masks, top)
    )
    objective = (
        ENTIRE_SPACE_CASCADE
        if config.world_version == "creator-neural-feed-supply-v4"
        else MASKED_CONDITIONAL
    )
    first, second = int(config.requests * 0.70), int(config.requests * 0.85)
    mean = exposed_features[:first].flatten(0, 1).mean(0)
    scale = exposed_features[:first].flatten(0, 1).std(0).clamp_min(1e-4)
    flat_labels = exposed_labels[:first].flatten(0, 1)
    flat_masks = exposed_masks[:first].flatten(0, 1)
    positives = (flat_labels * flat_masks).sum(0).clamp_min(1.0)
    observed = flat_masks.sum(0)
    positive_weight = ((observed - positives) / positives).clamp(max=30.0)
    if objective == ENTIRE_SPACE_CASCADE:
        entire_labels, entire_masks = entire_space_targets(
            exposed_labels[:first], exposed_masks[:first]
        )
        flat_labels = entire_labels.flatten(0, 1)
        flat_masks = entire_masks.flatten(0, 1)
        positives = (flat_labels * flat_masks).sum(0).clamp_min(1.0)
        observed = flat_masks.sum(0)
        positive_weight = ((observed - positives) / positives).clamp(max=30.0)
        positive_weight[:3] = 1.0
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
                loss = multitask_loss(
                    outputs, exposed_labels[request],
                    exposed_masks[request], positive_weight, objective,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        offline = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "final_loss": float(np.mean(losses[-max(first // config.train_batch_requests, 1):])),
            "metrics": _offline(
                model, exposed_features[first:second], exposed_semantic[first:second],
                history[first:second], exposed_labels[first:second],
                exposed_masks[first:second], mean, scale,
                objective,
            ),
            "objective": objective,
            "probability_space": {
                "click": "P(click|impression)",
                "create": (
                    "P(click_and_create|impression)"
                    if objective == ENTIRE_SPACE_CASCADE
                    else "P(create|click)"
                ),
                "publish": (
                    "P(click_and_create_and_publish|impression)"
                    if objective == ENTIRE_SPACE_CASCADE
                    else "P(publish|create)"
                ),
                "quality": "P(quality|publish)",
                "risk": "P(risk|publish)",
            },
        }
        offsets = {
            task: (
                torch.zeros_like(positive_weight[index])
                if objective == ENTIRE_SPACE_CASCADE and index < 3
                else positive_weight[index].log()
            )
            for index, task in enumerate(FEED_POSTING_TASKS)
        }
        offline["weighted_loss_logit_offsets"] = {
            task: float(offset) for task, offset in offsets.items()
        }
        bundles[name] = FeedPostingBundle(
            name, model, mean, scale, offsets, offline, objective
        )
    return bundles


def save_bundle(bundle, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "feed-posting-model-v3", "name": bundle.name,
        "feature_width": len(bundle.mean), "semantic_dim": config.semantic_dim,
        "state_dict": bundle.model.state_dict(), "mean": bundle.mean,
        "scale": bundle.scale, "logit_offsets": bundle.logit_offsets,
        "offline": bundle.offline,
        "objective": bundle.objective,
    }, path)
    return {"artifact_file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_bundle(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") not in {
        "feed-posting-model-v2", "feed-posting-model-v3"
    }:
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
        payload["offline"], payload.get("objective", MASKED_CONDITIONAL),
    )
