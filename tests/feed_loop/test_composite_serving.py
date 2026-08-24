from __future__ import annotations

import unittest

import torch

from fid_lab.feed_loop.serving.contracts import (
    CandidateScoreBundle,
    CompositeValueTreeConfig,
)
from fid_lab.feed_loop.serving.composite import CompositeTensorPolicy
from fid_lab.feed_loop.serving.launch import _decision
from fid_lab.feed_loop.serving.value_tree import CompositeValueTree


class _FeedPolicy:
    blend_weight = 0.01
    base_tolerance = 0.05
    name = "mmoe_guarded_0.01"
    model_name = "mmoe"

    def describe(self):
        return {"name": self.name}

    def value(self, predictions, features):
        del features
        return (
            0.55 * predictions["stay_norm"]
            + 0.18 * predictions["quality_long_view"]
            - 0.35 * predictions["negative_feedback"]
        )

    platform_value = value


class CompositeServingTest(unittest.TestCase):
    def test_composite_policy_rejects_release_weight_drift(self):
        local = type("Local", (), {"name": "linear"})()
        with self.assertRaisesRegex(ValueError, "blend differ"):
            CompositeTensorPolicy(
                _FeedPolicy(), local,
                CompositeValueTreeConfig(feed_residual_weight=0.02),
            )

    def test_value_tree_combines_models_only_for_eligible_business_candidates(self):
        shape = (2, 3)
        feed = {
            name: torch.full(shape, 0.2)
            for name in (
                "play_3s", "stay_norm", "completion", "long_view",
                "quality_long_view", "like", "negative_feedback",
                "anchor_click", "conversion", "returned_next_session",
            )
        }
        local = {
            name: torch.full(shape, 0.4)
            for name in (
                "anchor_click", "poi_detail", "poi_favorite", "conversion",
                "negative_feedback", "stay_norm",
            )
        }
        bundle = CandidateScoreBundle(
            torch.zeros(shape), feed, local,
            {"ad": torch.zeros(shape), "live": torch.zeros(shape)},
            {"feed": "mmoe", "local": "linear", "value_tree": "v1"},
        )
        candidates = {"is_poi": torch.tensor([[0, 1, 0], [1, 0, 1]])}
        tree = CompositeValueTree(
            _FeedPolicy(), CompositeValueTreeConfig(feed_residual_weight=0.0)
        )
        score, components = tree.evaluate(
            bundle, torch.zeros(2, 3, 28), candidates
        )
        self.assertTrue(torch.all(
            components["local_model_value"][candidates["is_poi"] == 0] == 0
        ))
        self.assertTrue(torch.all(
            score[candidates["is_poi"] == 1]
            > score[candidates["is_poi"] == 0].max()
        ))

    def test_bundle_rejects_missing_primitive_heads(self):
        bundle = CandidateScoreBundle(
            torch.zeros(1, 2), {}, {},
            {"ad": torch.zeros(1, 2), "live": torch.zeros(1, 2)}, {},
        )
        with self.assertRaisesRegex(ValueError, "Feed primitive"):
            bundle.validate()

    def test_shadow_pass_does_not_claim_randomized_ab_launch(self):
        shadow = {"primary": True, "lt": True}
        online = {
            "local_primary_positive": True,
            "platform_lt_positive": False,
            "stay_nonnegative": False,
            "negative_nonpositive": True,
        }
        self.assertEqual(
            _decision(shadow, online),
            "continue_powered_online_experiment",
        )


if __name__ == "__main__":
    unittest.main()
