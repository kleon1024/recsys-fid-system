"""Central tensor Value Tree for Feed and business model predictions."""

from __future__ import annotations

import torch

from .contracts import CandidateScoreBundle, CompositeValueTreeConfig
from ..scale.model_ladder.v4.serving import TensorV4RequestPolicy


LOCAL_ORDER = (
    "anchor_click", "poi_detail", "poi_favorite", "conversion",
    "stay_norm", "negative_feedback",
)


def request_standardize(values):
    return (values - values.mean(dim=1, keepdim=True)) / (
        values.std(dim=1, keepdim=True).clamp_min(1e-4)
    )


class CompositeValueTree:
    def __init__(
        self, feed_policy: TensorV4RequestPolicy,
        config=CompositeValueTreeConfig(),
    ):
        self.feed_policy = feed_policy
        self.config = config

    def evaluate(self, bundle: CandidateScoreBundle, features, candidates):
        bundle.validate()
        feed_value = self.feed_policy.platform_value(
            bundle.feed_predictions, features
        )
        feed_residual = request_standardize(feed_value)
        local_value = self.local_value(bundle.local_predictions, candidates)

        queue_value = (
            self.config.ad_value_weight * bundle.queue_values["ad"]
            + self.config.live_value_weight * bundle.queue_values["live"]
        )
        feed_score = (
            bundle.base_feed_score
            + self.config.feed_residual_weight * feed_residual
        )
        final = (
            feed_score + self.config.local_fine_weight * local_value
            + queue_value
        )
        return final, {
            "feed_model_value": feed_value,
            "feed_served_score": feed_score,
            "local_model_value": local_value,
            "queue_value": queue_value,
            "value_tree_score": final,
        }

    def local_value(self, predictions, candidates):
        local_value = torch.zeros_like(predictions["anchor_click"])
        for name, weight in zip(LOCAL_ORDER, self.config.local_weights, strict=True):
            local_value += weight * predictions[name]
        local_value *= candidates["is_poi"]
        return local_value
