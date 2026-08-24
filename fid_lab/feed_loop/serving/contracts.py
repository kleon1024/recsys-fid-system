"""Typed scores exchanged by Feed and business-specific rankers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class CandidateScoreBundle:
    base_feed_score: torch.Tensor
    feed_predictions: dict[str, torch.Tensor]
    local_predictions: dict[str, torch.Tensor]
    queue_values: dict[str, torch.Tensor]
    model_versions: dict[str, str]

    def validate(self) -> None:
        shape = self.base_feed_score.shape
        groups = (
            self.feed_predictions, self.local_predictions, self.queue_values
        )
        if self.base_feed_score.ndim != 2:
            raise ValueError("candidate score bundle must be request by candidate")
        for values in groups:
            if any(value.shape != shape for value in values.values()):
                raise ValueError("candidate score bundle has inconsistent shapes")
        required_feed = {
            "play_3s", "stay_norm", "completion", "long_view",
            "quality_long_view", "like", "negative_feedback",
            "anchor_click", "conversion", "returned_next_session",
        }
        required_local = {
            "anchor_click", "poi_detail", "poi_favorite", "conversion",
            "negative_feedback", "stay_norm",
        }
        if not required_feed.issubset(self.feed_predictions):
            raise ValueError("Feed primitive scores are incomplete")
        if not required_local.issubset(self.local_predictions):
            raise ValueError("Local primitive scores are incomplete")


@dataclass(frozen=True)
class CompositeValueTreeConfig:
    version: str = "composite-feed-business-value-tree-v1"
    feed_residual_weight: float = 0.01
    base_tolerance: float = 0.05
    local_coarse_weight: float = 0.025
    local_coarse_keep: int = 20
    local_fine_weight: float = 0.025
    ad_value_weight: float = 0.0
    live_value_weight: float = 0.0
    local_weights: tuple[float, ...] = (
        0.25, 0.20, 0.10, 0.25, 0.10, -0.20,
    )

    def __post_init__(self):
        if min(
            self.feed_residual_weight,
            self.local_coarse_weight,
            self.local_fine_weight,
            self.ad_value_weight,
            self.live_value_weight,
        ) < 0.0:
            raise ValueError("composite Value Tree weights must be nonnegative")
        if self.local_coarse_keep < 1:
            raise ValueError("composite coarse budget must be positive")
        if len(self.local_weights) != 6:
            raise ValueError("Local Value Tree requires six primitive weights")

    def manifest(self) -> dict[str, object]:
        return asdict(self)
