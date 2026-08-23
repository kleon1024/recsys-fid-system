"""POI distribution task, model, and training contracts."""

from __future__ import annotations

from dataclasses import dataclass


TASK_LABELS = {
    "anchor_click": 9,
    "poi_detail": 10,
    "poi_favorite": 11,
    "conversion": 12,
    "negative_feedback": 8,
    "stay_norm": 2,
}
MODEL_NAMES = ("linear", "wide_deep", "dcnv2", "mmoe")


@dataclass(frozen=True)
class PoiDistributionTrainingConfig:
    feature_dim: int = 28
    epochs: int = 4
    batch_size: int = 8_192
    learning_rate: float = 1.5e-3
    seed: int = 20260824
    device: str = "cuda:0"

    def __post_init__(self):
        if self.feature_dim != 28:
            raise ValueError("POI distribution requires the canonical 28 features")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("training sizes must be positive")
