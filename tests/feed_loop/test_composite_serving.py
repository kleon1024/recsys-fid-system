from __future__ import annotations

import unittest

import torch

from fid_lab.feed_loop.scale.experiment.trigger import combine_tensor_cuped_ab
from fid_lab.feed_loop.scale.graph.reporting import CELL_METRICS
from fid_lab.feed_loop.serving.contracts import (
    CandidateScoreBundle,
    CompositeValueTreeConfig,
)
from fid_lab.feed_loop.serving.aggregate import aggregate_composite_launches
from fid_lab.feed_loop.serving.composite import CompositeTensorPolicy
from fid_lab.feed_loop.serving.launch import _control_policy, _decision
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
    def test_preperiod_cuped_recovers_effect_and_reduces_variance(self):
        users = 20_000
        generator = torch.Generator().manual_seed(17)
        pre = torch.randn(users, len(CELL_METRICS), generator=generator)
        noise = 0.15 * torch.randn(
            users, len(CELL_METRICS), generator=generator
        )
        control = 2.0 * pre + noise
        treatment = control + 0.01
        left = {
            "config": {"experiment_salt": 0x1B873593},
            "_all_user_metrics": control,
            "_preperiod_user_metrics": pre,
        }
        right = {
            "config": {"experiment_salt": 0x1B873593},
            "_all_user_metrics": treatment,
            "_preperiod_user_metrics": pre.clone(),
        }
        metric = combine_tensor_cuped_ab(left, right)["lt_value_per_user"]
        self.assertLess(metric["standard_error"], 0.003)
        self.assertGreater(metric["variance_reduction"], 0.95)
        self.assertLess(metric["confidence_interval"][0], 0.01)
        self.assertGreater(metric["confidence_interval"][1], 0.01)
        self.assertEqual(metric["preperiod_max_abs_delta"], 0.0)

    def test_incremental_launch_uses_composite_model_as_control(self):
        class _Bundle:
            def __init__(self, name):
                self.name = name

        feed = _FeedPolicy()
        config = CompositeValueTreeConfig(feed_residual_weight=0.01)
        control = _control_policy(feed, _Bundle("linear"), config)
        treatment = _control_policy(feed, _Bundle("mmoe"), config)
        self.assertIn("linear", control.name)
        self.assertIn("mmoe", treatment.name)
        split = CompositeTensorPolicy(
            feed, _Bundle("mmoe"), config, _Bundle("linear")
        )
        self.assertEqual(split.describe()["local_coarse_model"], "linear")
        self.assertEqual(split.describe()["local_fine_model"], "mmoe")

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
            "platform_lt_direction_nonnegative": False,
            "platform_lt_noninferior": True,
            "stay_noninferior": True,
            "negative_guardrail": True,
        }
        self.assertEqual(
            _decision(shadow, online),
            "continue_powered_online_experiment",
        )

    def test_multi_salt_aggregate_rejects_oracle_as_online_gate(self):
        names = tuple(CELL_METRICS)

        def metrics(delta, oracle_delta):
            def effect(name):
                if name == "coarse_feed_oracle_recall":
                    return oracle_delta
                if name == "negative_rate":
                    return -delta
                return delta

            return {
                name: {
                    "control_mean": 1.0,
                    "treatment_mean": 1.0 + effect(name),
                    "standard_error": 0.001,
                    "confidence_interval": [
                        effect(name) - 0.00196,
                        effect(name) + 0.00196,
                    ],
                }
                for name in names
            }

        reports = []
        for salt in (11, 23, 47):
            reports.append({
                "schema": "unified-feed-business-serving-launch-v3",
                "config": {"experiment_salt": salt, "users": 10_000},
                "control": {"name": "linear"},
                "treatment": {"name": "mmoe"},
                "behavior_world": {"authority": "v4"},
                "paired_shadow_replay": metrics(0.01, 0.01),
                "online_cuped_ab": metrics(0.01, -0.5),
            })
        pooled = aggregate_composite_launches(reports)
        self.assertEqual(pooled["decision"], "pass")
        self.assertNotIn("coarse_recall_noninferior", pooled["online_gates"])
        self.assertLess(
            pooled["oracle_diagnostics"]["coarse_feed_oracle_recall"][
                "confidence_interval"
            ][1],
            0,
        )


if __name__ == "__main__":
    unittest.main()
