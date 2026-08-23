"""Segment-aware fusion of calibrated primitive predictions for Feed ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PredictedFeedValueConfig:
    version: str = "feed-predicted-value-tree-v2"
    mature: tuple[float, ...] = (
        0.08, 0.55, 0.03, 0.10, 0.18, 0.06, -0.35, 0.00,
    )
    cold: tuple[float, ...] = (
        0.16, 0.45, 0.08, 0.08, 0.12, 0.04, -0.35, 0.00,
    )
    stay_completion_cross: float = 0.03
    quality_interaction_cross: float = 0.04
    fatigue_negative_cross: float = -0.06
    local_anchor_weight: float = 0.04
    local_conversion_weight: float = 0.02

    def manifest(self) -> dict[str, object]:
        return asdict(self)


TASK_ORDER = (
    "play_3s", "stay_norm", "completion", "long_view",
    "quality_long_view", "like", "negative_feedback",
    "returned_next_session",
)


def predicted_feed_value(tasks, features, config=PredictedFeedValueConfig()):
    """Fuse predictions; LT remains an experiment outcome outside this tree."""
    cold_share = (features[:, 26] <= (1.0 / 3.0)).float()
    score = 0.0
    for index, task in enumerate(TASK_ORDER):
        weight = (
            config.mature[index] * (1.0 - cold_share)
            + config.cold[index] * cold_share
        )
        score = score + weight * tasks[task]
    score = score + config.stay_completion_cross * (
        tasks["stay_norm"] * tasks["completion"]
    )
    score = score + config.quality_interaction_cross * (
        tasks["quality_long_view"] * tasks["like"]
    )
    score = score + config.fatigue_negative_cross * (
        features[:, 7] * tasks["negative_feedback"]
    )
    score = score + features[:, 13] * (
        config.local_anchor_weight * tasks["anchor_click"]
        + config.local_conversion_weight * tasks["conversion"]
    )
    return score
