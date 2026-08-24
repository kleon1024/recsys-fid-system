"""IPS-corrected entire-space training and artifact replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from torch import nn

from .architectures import build_ranker
from .evaluation import calibration_biases, metrics
from .objectives import (
    multi_task_loss,
    positive_weights as compute_positive_weights,
    task_probabilities,
)
from ..contracts import MODEL_NAMES, TASK_LABELS
from ..data import PoiDistributionSplit


@dataclass(frozen=True)
class PoiRankerBundle:
    name: str
    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    calibration_biases: dict[str, float]
    offline: dict[str, object]

    def probabilities(self, features, chunk=250_000):
        output = {task: [] for task in TASK_LABELS}
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(features), chunk):
                logits = self.model(
                    (features[start:start + chunk] - self.mean) / self.scale
                )
                for task, value in task_probabilities(
                    logits, self.calibration_biases
                ).items():
                    output[task].append(value)
        return {task: torch.cat(parts) for task, parts in output.items()}

    def score(self, features, chunk=250_000):
        probability = self.probabilities(features, chunk)
        return (
            0.45 * probability["anchor_click"]
            + 0.25 * probability["poi_detail"]
            + 0.05 * probability["poi_favorite"]
            + 0.15 * probability["conversion"]
            + 0.05 * probability["stay_norm"]
            - 0.20 * probability["negative_feedback"]
        )


def _listwise_epoch(model, split, mean, scale, optimizer, config, generator):
    requests = len(split.positive_candidate_features)
    order = torch.randperm(requests, generator=generator)
    losses = []
    request_batch = max(config.batch_size // 16, 64)
    for start in range(0, requests, request_batch):
        index = order[start:start + request_batch]
        candidates = split.positive_candidate_features[index].to(mean.device)
        shape = candidates.shape[:2]
        output = model(((candidates - mean) / scale).flatten(0, 1))
        logits = output["anchor_click"].reshape(shape)
        target = split.positive_candidate_index[index].to(mean.device)
        point = nn.functional.cross_entropy(logits, target, reduction="none")
        weights = split.positive_candidate_weights[index].to(mean.device)
        loss = 0.15 * (point * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def train_rankers(config, train: PoiDistributionSplit, validation):
    device = torch.device(config.device)
    mean = train.features.mean(0).to(device)
    scale = train.features.std(0).clamp_min(1e-4).to(device)
    positive_weight_by_task = {
        task: value.to(device)
        for task, value in compute_positive_weights(train.labels).items()
    }
    bundles = {}
    for offset, name in enumerate(MODEL_NAMES):
        torch.manual_seed(config.seed + offset)
        model = build_ranker(name, config.feature_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        generator = torch.Generator().manual_seed(config.seed + 100 + offset)
        losses = []
        for _ in range(config.epochs):
            order = torch.randperm(len(train), generator=generator)
            for start in range(0, len(order), config.batch_size):
                index = order[start:start + config.batch_size]
                features = train.features[index].to(device)
                labels = train.labels[index].to(device)
                weights = train.weights[index].to(device)
                outputs = model((features - mean) / scale)
                loss = multi_task_loss(
                    outputs, labels, weights, positive_weight_by_task
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            losses.extend(_listwise_epoch(
                model, train, mean, scale, optimizer, config, generator
            ))
        biases = calibration_biases(model, mean, scale, validation, device)
        bundle = PoiRankerBundle(
            name, model, mean, scale, biases,
            {
                "parameters": sum(value.numel() for value in model.parameters()),
                "loss_history_tail": losses[-10:],
                "calibration_biases": biases,
                "listwise_positive_requests": len(
                    train.positive_candidate_features
                ),
            },
        )
        bundle.offline["validation"] = metrics(bundle, validation)
        bundles[name] = bundle
    return bundles


def save_bundle(bundle, path: Path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "poi-distribution-ranker-v1",
        "name": bundle.name,
        "config": asdict(config),
        "state_dict": bundle.model.state_dict(),
        "mean": bundle.mean,
        "scale": bundle.scale,
        "calibration_biases": bundle.calibration_biases,
        "offline": bundle.offline,
    }, path)
    return {"artifact_file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_bundle(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "poi-distribution-ranker-v1":
        raise ValueError("unsupported POI distribution artifact")
    model = build_ranker(payload["name"], payload["config"]["feature_dim"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return PoiRankerBundle(
        payload["name"], model, payload["mean"].to(device),
        payload["scale"].to(device), payload["calibration_biases"],
        payload["offline"],
    )
