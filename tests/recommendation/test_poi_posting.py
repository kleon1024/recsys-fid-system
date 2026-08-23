from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fid_lab.poi_posting import PoiPostingConfig, PoiPostingRanker, build_dataset
from fid_lab.poi_posting.training import sampled_training_indices, tensor_batch, time_split
from fid_lab.poi_posting.world import PostingWorldConfig, run_posting_launch_ladder
from fid_lab.poi_posting.world.generator import (
    build_world,
    candidate_features,
    retrieve,
    rule_score,
    simulate_response,
)
from fid_lab.simulation.local import (
    SupplySwitchbackConfig,
    calibrate_supply_switchback,
    run_supply_switchback,
)
from fid_lab.value import DEFAULT_LT_CONFIG


class PoiPostingTest(unittest.TestCase):
    def test_teacher_hidden_posting_world_closes_request_candidates(self) -> None:
        config = PostingWorldConfig(
            requests=500, cities=8, categories=4, items_per_cell=32,
            semantic_dim=8, train_epochs=1, device="cpu",
        )
        world = build_world(config)
        candidates = retrieve(world, ("popular", "geo"))
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))
        self.assertEqual(tuple(candidates.item_ids.shape), (500, 20))
        self.assertLess(float(candidates.audit_oracle_recalled.float().mean()), 0.5)
        self.assertTrue(torch.all(response["labels"][:, :, 0].sum(dim=1) <= 1))
        self.assertTrue(torch.all(
            response["labels"][:, :, 1] <= response["labels"][:, :, 0]
        ))
        self.assertEqual(tuple(response["top_indices"].shape), (500, 8))

    def test_posting_launch_ladder_uses_last_accepted_controls(self) -> None:
        with TemporaryDirectory() as directory:
            report = run_posting_launch_ladder(PostingWorldConfig(
                requests=1_000, cities=8, categories=4, items_per_cell=32,
                semantic_dim=8, train_epochs=1, train_batch_pairs=512,
                device="cpu",
            ), Path(directory))
            for model in report["models"].values():
                self.assertEqual(model["serialized_replay_max_abs_delta"], 0.0)
                self.assertTrue(
                    (Path(directory) / model["artifact"]["artifact_file"]).exists()
                )
        self.assertFalse(
            report["logging_contract"]["oracle_forced_into_candidates"]
        )
        self.assertEqual(
            {row["stage"] for row in report["launches"]},
            {"candidate", "fine", "end_to_end"},
        )
        active = {"candidate": "popular_geo", "fine": "rule"}
        for row in report["launches"][:-1]:
            expected = (
                "popular_geo" if row["stage"] == "candidate"
                else active[row["stage"]]
            )
            self.assertEqual(row["control"], expected)
            if row["promoted"] and row["stage"] == "fine":
                active[row["stage"]] = row["treatment"]

    def test_supply_switchback_uses_platform_metrics_for_lt(self) -> None:
        config = SupplySwitchbackConfig(cities=20, periods=12, users_per_city_period=100)
        report = run_supply_switchback(config)
        metrics = report["metrics"]
        expected = (
            metrics["stay_seconds_per_user"]["known_dgp_effect"]
            / 60.0
            * DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
            + metrics["active_days_per_user"]["known_dgp_effect"]
            * DEFAULT_LT_CONFIG.rates["active_day"].unit_value
        )
        self.assertAlmostEqual(
            metrics["lt_value_per_user"]["known_dgp_effect"], expected
        )
        self.assertGreater(
            metrics["local_commercialization_per_user"]["known_dgp_effect"], 0.0
        )
        self.assertLess(
            report["effective_user_periods"],
            config.cities * config.periods * config.users_per_city_period,
        )

    def test_switchback_calibration_reuses_exact_estimator(self) -> None:
        report = calibrate_supply_switchback(
            SupplySwitchbackConfig(cities=12, periods=10, users_per_city_period=100),
            simulations=4,
        )
        self.assertIn("two-way fixed effects", report["estimator"])
        coverage = report["metrics"]["lt_value_per_user"]["coverage_rate"]
        self.assertGreaterEqual(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PoiPostingConfig(sessions=80, epochs=1)
        cls.data = build_dataset(cls.config)

    def test_contract_preserves_one_slate_per_session(self) -> None:
        counts = np.bincount(self.data.session_id)
        self.assertTrue(np.all(counts == self.config.candidates_per_session))
        self.assertGreater(int(self.data.hard_negative.sum()), 0)
        self.assertTrue(np.all(self.data.label_masks == 1.0))

    def test_easy_negative_sampling_keeps_all_important_examples(self) -> None:
        train, _, _ = time_split(self.data)
        selected, weights = sampled_training_indices(self.data, train, self.config)
        required = train[
            (self.data.labels[train, 0] > 0) | (self.data.hard_negative[train] > 0)
        ]
        self.assertTrue(set(required).issubset(set(selected)))
        self.assertEqual(len(selected), len(weights))

    def test_model_outputs_all_tasks_and_frame_attention(self) -> None:
        model = PoiPostingRanker(self.config)
        outputs = model(tensor_batch(self.data, np.arange(12)))
        for task in ("select", "publish", "relevance"):
            self.assertEqual(tuple(outputs[task].shape), (12,))
            gate = outputs[f"gate:{task}"].detach().numpy()
            np.testing.assert_allclose(gate.sum(axis=1), 1.0, atol=1e-6)
        attention = self.data.frame_attention[:12]
        np.testing.assert_allclose(attention.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
