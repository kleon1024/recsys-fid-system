"""Request-level views over the frozen V4 Feed logging authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RequestSplit:
    tensors: dict[str, torch.Tensor]
    indices: torch.Tensor
    ips_clip: float = 60.0
    resident: dict[str, torch.Tensor] | None = None
    weight_normalizer: float = 1.0

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def feature_dim(self) -> int:
        return self.tensors["candidate_features"].shape[-1]

    @property
    def sequence_dim(self) -> int:
        return self.tensors["behavior_sequence"].shape[-1]

    @property
    def candidate_count(self) -> int:
        return self.tensors["candidate_features"].shape[1]

    def selected_features(self) -> np.ndarray:
        index = self.indices
        exposed = self.tensors["exposed_index"][index].long()
        return self.tensors["candidate_features"][index, exposed].float().numpy()

    def labels(self) -> np.ndarray:
        return self.tensors["labels"][self.indices].float().numpy()

    def masks(self) -> np.ndarray:
        return self.tensors["label_masks"][self.indices].bool().numpy()

    def user_ids(self) -> np.ndarray:
        return self.tensors["user_id"][self.indices].long().numpy()

    def weights(self) -> np.ndarray:
        propensity = self.tensors["exposure_propensity"][self.indices].float()
        weights = propensity.reciprocal().clamp_max(self.ips_clip)
        return (weights / max(self.weight_normalizer, 1e-6)).numpy()

    def batch(self, local_indices: torch.Tensor, device: torch.device):
        if self.resident is not None:
            index = local_indices.to(device)
            propensity = self.resident["propensity"][index]
            weights = propensity.reciprocal().clamp_max(self.ips_clip)
            weights /= max(self.weight_normalizer, 1e-6)
            return {
                "candidates": self.resident["candidates"][index],
                "sequence": self.resident["sequence"][index],
                "exposed_index": self.resident["exposed_index"][index],
                "labels": self.resident["labels"][index],
                "masks": self.resident["masks"][index],
                "weights": weights,
            }
        index = self.indices[local_indices]
        propensity = self.tensors["exposure_propensity"][index].float()
        weights = propensity.reciprocal().clamp_max(self.ips_clip)
        weights /= max(self.weight_normalizer, 1e-6)
        return {
            "candidates": self.tensors["candidate_features"][index].float().to(device),
            "sequence": self.tensors["behavior_sequence"][index].float().to(device),
            "exposed_index": self.tensors["exposed_index"][index].long().to(device),
            "labels": self.tensors["labels"][index].float().to(device),
            "masks": self.tensors["label_masks"][index].bool().to(device),
            "weights": weights.to(device),
        }

    def stage(self, device: torch.device) -> None:
        index = self.indices
        self.resident = {
            "candidates": self.tensors["candidate_features"][index].float().to(device),
            "sequence": self.tensors["behavior_sequence"][index].float().to(device),
            "exposed_index": self.tensors["exposed_index"][index].long().to(device),
            "labels": self.tensors["labels"][index].float().to(device),
            "masks": self.tensors["label_masks"][index].bool().to(device),
            "propensity": self.tensors["exposure_propensity"][index].float().to(device),
        }


def load_request_split(
    dataset_dir: Path,
    split: str,
    limit: int | None = None,
    seed: int = 20260823,
) -> RequestSplit:
    payload = torch.load(
        dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False
    )
    tensors = payload["tensors"]
    count = len(tensors["request_id"])
    indices = torch.arange(count)
    if limit is not None and limit < count:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(count, generator=generator)[:limit].sort().values
    result = RequestSplit(tensors=tensors, indices=indices)
    propensity = tensors["exposure_propensity"][indices].float()
    result.weight_normalizer = float(
        propensity.reciprocal().clamp_max(result.ips_clip).mean()
    )
    return result
