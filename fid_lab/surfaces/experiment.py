"""Deterministic executable checks for every configured recommendation surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as functional

from .contracts import SURFACE_SPECS, SurfaceSpec
from .model import build_surface_model


@dataclass(frozen=True)
class SurfaceReport:
    surface: str
    examples: int
    task_auc: dict[str, float]
    positive_rate: dict[str, float]
    mean_gate_entropy: dict[str, float]


def make_dataset(
    spec: SurfaceSpec, examples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(examples, len(spec.features))).astype(np.float32)
    labels = np.zeros((examples, len(spec.tasks)), dtype=np.float32)
    feature_index = {name: index for index, name in enumerate(spec.features)}
    shared_intent = rng.normal(0.0, 0.5, size=examples)
    for task_index, task in enumerate(spec.tasks):
        logits = np.full(examples, task.base_logit) + 0.25 * shared_intent
        for feature_name, weight in task.drivers:
            logits += weight * features[:, feature_index[feature_name]]
        probability = 1.0 / (1.0 + np.exp(-logits))
        labels[:, task_index] = rng.binomial(1, probability)
    sequence = None
    if spec.name == "feed_poi_video":
        sequence = rng.normal(size=(examples, 24, 8)).astype(np.float32)
        affinity = spec.features.index("sequence_affinity")
        sequence[:, :, 0] += features[:, affinity, None]
    return features, labels, sequence


def train_surface(
    spec: SurfaceSpec, examples: int = 3_000, epochs: int = 12
) -> SurfaceReport:
    seed = sum(ord(value) for value in spec.name)
    features, labels, sequence = make_dataset(spec, examples, seed)
    split = int(examples * 0.8)
    torch.manual_seed(seed)
    model = build_surface_model(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    train_x = torch.from_numpy(features[:split])
    train_y = torch.from_numpy(labels[:split])
    train_sequence = None if sequence is None else torch.from_numpy(sequence[:split])
    positive = train_y.sum(dim=0).clamp_min(1.0)
    positive_weight = ((split - positive) / positive).clamp(max=20.0)
    rng = np.random.default_rng(seed + 1)
    for _ in range(epochs):
        order = rng.permutation(split)
        for start in range(0, split, 256):
            index = torch.from_numpy(order[start : start + 256])
            sequence_batch = None if train_sequence is None else train_sequence[index]
            outputs = model(train_x[index], sequence_batch)
            losses = [
                functional.binary_cross_entropy_with_logits(
                    outputs[task.name],
                    train_y[index, task_index],
                    pos_weight=positive_weight[task_index],
                )
                for task_index, task in enumerate(spec.tasks)
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    task_auc: dict[str, float] = {}
    positive_rate: dict[str, float] = {}
    gate_entropy: dict[str, float] = {}
    with torch.no_grad():
        test_sequence = None if sequence is None else torch.from_numpy(sequence[split:])
        outputs = model(torch.from_numpy(features[split:]), test_sequence)
    for task_index, task in enumerate(spec.tasks):
        scores = torch.sigmoid(outputs[task.name]).numpy()
        task_labels = labels[split:, task_index]
        task_auc[task.name] = float(roc_auc_score(task_labels, scores))
        positive_rate[task.name] = float(task_labels.mean())
        gate_key = f"gate:{task.name}"
        if gate_key in outputs:
            gate = outputs[gate_key].numpy()
            gate_entropy[task.name] = float(
                -(gate * np.log(np.maximum(gate, 1e-8))).sum(axis=1).mean()
            )
    return SurfaceReport(spec.name, examples, task_auc, positive_rate, gate_entropy)


def run_surface_suite() -> dict[str, object]:
    reports = [train_surface(spec) for spec in SURFACE_SPECS.values()]
    return {
        "surfaces": {report.surface: asdict(report) for report in reports},
        "all_tasks_above_random": all(
            min(report.task_auc.values()) > 0.55 for report in reports
        ),
    }
