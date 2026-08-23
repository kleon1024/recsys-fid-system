import unittest
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fid_lab.evolution.evaluation.metrics import grouped_auc
from fid_lab.feed_loop.scale.lt_exchange import combine_lt_exchange_sensitivity
from fid_lab.feed_loop.scale.tensor_engine import (
    LOCAL_STATIC,
    LOCAL_INTENT_RANKER,
    PERSONALIZED,
    TensorFeedConfig,
    _candidate_batch,
    _new_user_state,
    run_tensor_feed,
)
from fid_lab.feed_loop.scale.tensor_catalog import build_tensor_catalog
from fid_lab.feed_loop.scale.artifact.cli import _launch_decision
from fid_lab.feed_loop.scale.artifact.features import build_tensor_features
from fid_lab.feed_loop.models.feature_lr import (
    train_feature_lr_campaign,
    train_feature_lr_suite,
)
from fid_lab.feed_loop.scale.artifact.feature_lr_cli import run_feature_lr_launches
from fid_lab.simulation.features import campaign_candidate_sets


class GroupedAUCTest(unittest.TestCase):
    def test_feature_lr_launches_always_use_last_accepted_control(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = train_feature_lr_suite(80, 300, root / "artifacts", seed=31)
            report_path = root / "training.json"
            report_path.write_text(json.dumps(report))
            launches = run_feature_lr_launches(
                report_path,
                root / "artifacts",
                TensorFeedConfig(
                    users=40,
                    steps=2,
                    candidates=5,
                    batch_users=20,
                    device="cpu",
                    signal_version="heterogeneous-nonlinear-v2",
                ),
            )
        self.assertEqual(len(launches["launches"]), 4)
        self.assertEqual(
            launches["launches"][0]["added_features"],
            ["long_sequence_match", "short_sequence_match"],
        )
        self.assertTrue(
            all("unified_lt_exchange" in launch for launch in launches["launches"])
        )
        expected_control = "basic"
        for launch in launches["launches"]:
            self.assertEqual(launch["control"], expected_control)
            self.assertEqual(
                launch["promotion"]["prior_active_key"], expected_control
            )
            expected_control = launch["promotion"]["resulting_active_key"]
        self.assertEqual(
            launches["release_state"]["active_key"], expected_control
        )

    def test_small_lr_campaign_reuses_exact_active_artifact(self):
        candidates = campaign_candidate_sets("hash_content_split_v1")
        self.assertEqual(len(candidates), 8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir = root / "base-artifacts"
            base = train_feature_lr_suite(80, 300, base_dir, seed=31)
            base_report = root / "base.json"
            base_report.write_text(json.dumps(base))
            campaign_dir = root / "campaign-artifacts"
            campaign = train_feature_lr_campaign(
                "hash_content_split_v1",
                80,
                300,
                campaign_dir,
                seed=31,
                base_report_path=base_report,
                base_artifact_dir=base_dir,
            )
            campaign_report = root / "campaign.json"
            campaign_report.write_text(json.dumps(campaign))
            base_key = campaign["campaign"]["base_key"]
            active_artifact = campaign["offline"][base_key]["artifact_manifest"]
            prior_release = root / "release.json"
            prior_release.write_text(json.dumps({
                "active_control_key": base_key,
                "active_control_artifact": active_artifact,
                "rollback_key": "basic__realtime",
                "rollback_artifact": None,
                "promoted_by_launch": "F-LR-003",
            }))
            launches = run_feature_lr_launches(
                campaign_report,
                campaign_dir,
                TensorFeedConfig(
                    users=40,
                    steps=2,
                    candidates=5,
                    batch_users=20,
                    device="cpu",
                    signal_version="heterogeneous-nonlinear-v2",
                ),
                prior_release,
            )

        self.assertEqual(
            active_artifact["artifact_id"],
            base["offline"][base_key]["artifact_manifest"]["artifact_id"],
        )
        self.assertEqual(
            [launch["launch_id"] for launch in launches["launches"]],
            ["F-LR-005", "F-LR-006", "F-LR-007"],
        )
        self.assertEqual(launches["launches"][0]["control"], base_key)

    def test_tensor_v2_features_are_finite_and_match_stateful_width(self):
        config = TensorFeedConfig(
            users=4,
            steps=2,
            candidates=5,
            topics=4,
            catalog_items=120,
            device="cpu",
            signal_version="heterogeneous-nonlinear-v2",
        )
        generator = torch.Generator().manual_seed(config.seed)
        catalog = build_tensor_catalog(config, generator, torch.device("cpu"))
        user_ids = torch.arange(config.users)
        state = _new_user_state(
            config, PERSONALIZED, generator, torch.device("cpu"), user_ids
        )
        candidates = _candidate_batch(
            config, generator, torch.device("cpu"), state, catalog, step=0
        )
        features = build_tensor_features(config, user_ids, state, candidates, 0)
        self.assertEqual(features.shape, (4, 5, 24))
        self.assertTrue(torch.isfinite(features).all())

    def test_tensor_launch_uses_unified_lt_not_unexchanged_quality(self):
        def metric(lift, p, interval=(-0.001, 0.001)):
            return {
                "control_mean": 1.0,
                "treatment_mean": 1.0 + lift,
                "relative_lift": lift,
                "p_value": p,
                "confidence_interval": interval,
            }

        decision = _launch_decision(
            {"passed": True},
            {
                "negative_rate": metric(0.0, 1.0),
                "quality_long_view_rate": metric(-0.015, 0.01, (-0.02, -0.01)),
                "stay_per_exposure": metric(0.008, 0.001, (0.004, 0.012)),
                "lt_value_per_user": metric(0.002, 0.02, (0.0002, 0.0038)),
            },
        )
        self.assertEqual(decision, "pass_unified_lt_nonnegative")

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
