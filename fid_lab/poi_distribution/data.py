"""Point-in-time exposed examples from the request-level candidate authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class PoiDistributionSplit:
    features: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    user_ids: torch.Tensor
    positive_candidate_features: torch.Tensor
    positive_candidate_index: torch.Tensor
    positive_candidate_weights: torch.Tensor

    def __len__(self):
        return len(self.features)


def load_exposed_split(dataset_dir: Path, split: str) -> PoiDistributionSplit:
    payload = torch.load(
        dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False
    )["tensors"]
    rows = torch.arange(len(payload["exposed_index"]))
    choice = payload["exposed_index"].long()
    propensity = payload["exposure_propensity"].float().clamp_min(1e-4)
    weights = propensity.reciprocal().clamp_max(20.0)
    weights /= weights.mean()
    positive = payload["labels"][:, 9] > 0
    return PoiDistributionSplit(
        payload["candidate_features"][rows, choice].float(),
        payload["labels"].float(),
        weights,
        payload["user_id"].long(),
        payload["candidate_features"][positive].float(),
        choice[positive],
        weights[positive],
    )
