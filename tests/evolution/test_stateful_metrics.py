import unittest
from dataclasses import replace

import numpy as np
import torch

from fid_lab.evolution.evaluation.metrics import grouped_auc
from fid_lab.feed_loop.scale.lt_exchange import combine_lt_exchange_sensitivity
from fid_lab.feed_loop.scale.tensor_engine import (
    LOCAL_STATIC,
    LOCAL_INTENT_RANKER,
    TensorFeedConfig,
    _candidate_batch,
    _new_user_state,
    run_tensor_feed,
)
from fid_lab.feed_loop.scale.tensor_catalog import build_tensor_catalog


class GroupedAUCTest(unittest.TestCase):
    def test_reports_single_class_coverage(self):
        report = grouped_auc(
            np.asarray([0, 1, 1, 1]),
            np.asarray([0.1, 0.9, 0.3, 0.4]),
            np.asarray([10, 10, 20, 20]),
        )

        self.assertEqual(report["value"], 1.0)
        self.assertEqual(report["eligible_groups"], 1)
        self.assertEqual(report["eligible_group_rate"], 0.5)
        self.assertEqual(report["eligible_record_rate"], 0.5)

    def test_rejects_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            grouped_auc(np.asarray([0]), np.asarray([0.2, 0.3]), np.asarray([1]))

    def test_tensor_local_conversion_metric_closes_cell_and_overall_contract(self):
        report = run_tensor_feed(
            TensorFeedConfig(
                users=40,
                steps=2,
                candidates=5,
                batch_users=20,
                device="cpu",
            ),
            LOCAL_STATIC,
        )
        self.assertIn("conversion_rate", report["metrics"])
        self.assertIn("conversion_rate", report["experiment_cells"]["control"])
        expected = (
            report["metrics"]["closed_loop_payment_rate"]
            + report["metrics"]["open_loop_conversion_rate"]
        )
        self.assertAlmostEqual(report["metrics"]["conversion_rate"], expected)
        self.assertEqual(
            report["metrics"]["accepted_platform_commercialization_per_user"],
            0.0,
        )
        for cell in ("control", "treatment"):
            self.assertEqual(
                report["experiment_cells"][cell][
                    "accepted_platform_commercialization_per_user"
                ]["mean"],
                0.0,
            )
        sensitivity = combine_lt_exchange_sensitivity(report, report)
        self.assertEqual(
            sensitivity["0"]["absolute_lift"],
            sensitivity["1"]["absolute_lift"],
        )

    def test_stable_catalog_forces_search_and_retarget_candidate_identity(self):
        config = TensorFeedConfig(
            users=4,
            candidates=5,
            topics=4,
            catalog_items=120,
            device="cpu",
        )
        generator = torch.Generator().manual_seed(config.seed)
        catalog = build_tensor_catalog(config, generator, torch.device("cpu"))
        user_ids = torch.arange(config.users)
        state = _new_user_state(
            config, LOCAL_STATIC, generator, torch.device("cpu"), user_ids
        )
        state["retarget_item"] = torch.tensor([7, 11, 19, 23])
        candidates = _candidate_batch(
            config, generator, torch.device("cpu"), state, catalog, step=0
        )
        torch.testing.assert_close(candidates["item_ids"][:, 0], state["retarget_item"])
        torch.testing.assert_close(
            candidates["candidate_topic"][:, 1], state["search_topic"]
        )

    def test_reverse_holdout_measures_only_post_burn_in(self):
        config = TensorFeedConfig(
            users=40,
            steps=4,
            candidates=5,
            batch_users=20,
            device="cpu",
        )
        report = run_tensor_feed(
            config,
            LOCAL_INTENT_RANKER,
            policy_schedule=(LOCAL_INTENT_RANKER,) * 4,
            measurement_start_step=2,
        )
        self.assertEqual(report["measurement_start_step"], 2)
        self.assertLessEqual(report["metrics"]["exposures"], config.users * 2)
        self.assertEqual(len(report["policy_schedule"]), config.steps)

    def test_coarse_budget_reports_oracle_pass_through(self):
        config = TensorFeedConfig(
            users=40,
            steps=2,
            candidates=20,
            batch_users=20,
            device="cpu",
        )
        policy = replace(LOCAL_INTENT_RANKER, coarse_keep=5)
        report = run_tensor_feed(config, policy)
        self.assertAlmostEqual(report["metrics"]["coarse_pass_fraction"], 0.25)
        self.assertGreaterEqual(report["metrics"]["coarse_feed_oracle_recall"], 0.0)
        self.assertLessEqual(report["metrics"]["coarse_feed_oracle_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
