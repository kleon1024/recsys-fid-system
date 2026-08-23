"""Load exposed mature examples from the frozen V3 request log."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ExposedSplit:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    user_ids: np.ndarray
    rule_scores: np.ndarray
    lifecycle: np.ndarray
    region: np.ndarray


def load_tensors(dataset_dir: Path, split: str) -> dict[str, torch.Tensor]:
    payload = torch.load(dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False)
    return payload["tensors"]


def exposed_split(dataset_dir: Path, split: str) -> ExposedSplit:
    tensors = load_tensors(dataset_dir, split)
    rows = torch.arange(len(tensors["exposed_index"]))
    choice = tensors["exposed_index"].long()
    features = tensors["candidate_features"][rows, choice].float().numpy()
    rule_scores = tensors["candidate_fine_scores"][rows, choice].float().numpy()
    propensity = tensors["exposure_propensity"].float().numpy()
    return ExposedSplit(
        features=features,
        labels=tensors["labels"].float().numpy(),
        weights=1.0 / np.maximum(propensity, 1e-4),
        user_ids=tensors["user_id"].long().numpy(),
        rule_scores=rule_scores,
        lifecycle=tensors["lifecycle_bucket"].long().numpy(),
        region=tensors["region_bucket"].long().numpy(),
    )
