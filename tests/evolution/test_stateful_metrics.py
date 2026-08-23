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
    combine_tensor_trigger_ab,
    run_tensor_feed,
)
from fid_lab.feed_loop.scale.graph.random import uniform
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
    def test_counter_rng_event_streams_are_decorrelated(self):
        user_ids = torch.arange(20_000)
        draws = torch.stack(
            [uniform(user_ids, 3, stream, 20260823) for stream in (31, 32, 34, 35)]
        )
        correlations = torch.corrcoef(draws)
        off_diagonal = correlations[~torch.eye(4, dtype=torch.bool)]
        self.assertLess(float(off_diagonal.abs().max()), 0.03)
        self.assertLess(float((draws.mean(dim=1) - 0.5).abs().max()), 0.01)

    def test_search_trigger_cohort_is_sparse_and_projects_to_all_users(self):
        config = TensorFeedConfig(
            users=2_000,
            steps=4,
            candidates=5,
            batch_users=500,
            device="cpu",
            search_event_rate=0.10,
            search_ttl_requests=2,
        )
        report = run_tensor_feed(
            config,
            LOCAL_STATIC,
            measurement_start_step=1,
            trigger_kind="post_search",
        )
        trigger = report["trigger_experiment"]
        combined = combine_tensor_trigger_ab(report, report)

        self.assertGreater(trigger["eligible_rate"], 0.0)
        self.assertLess(trigger["eligible_rate"], 0.5)
        self.assertEqual(
            sum(cell["lt_value_per_user"]["users"] for cell in trigger["cells"].values()),
            trigger["eligible_users"],
        )
        eligible = combined["eligible_ab"]["lt_value_per_user"]
        projected = combined["projected_overall_ab"]["lt_value_per_user"]
        self.assertAlmostEqual(
            projected["treatment_mean"] - projected["control_mean"],
            (eligible["treatment_mean"] - eligible["control_mean"])
            * trigger["eligible_rate"],
        )

    def test_pre_treatment_state_uses_burn_policy_not_challenger(self):
        config = TensorFeedConfig(
            users=1_000,
            steps=4,
            candidates=8,
            batch_users=250,
            catalog_items=2_000,
            device="cpu",
        )
        challenger = replace(LOCAL_INTENT_RANKER, observation_noise=0.65)
        control = run_tensor_feed(
            config,
            LOCAL_STATIC,
            policy_schedule=(LOCAL_STATIC,) * 4,
            measurement_start_step=2,
            trigger_kind="retarget",
        )
        treatment = run_tensor_feed(
            config,
            challenger,
            policy_schedule=(LOCAL_STATIC,) * 2 + (challenger,) * 2,
            measurement_start_step=2,
            trigger_kind="retarget",
        )
        self.assertEqual(
            control["trigger_experiment"]["eligible_users"],
            treatment["trigger_experiment"]["eligible_users"],
        )

    def test_counter_rng_is_invariant_to_gpu_batch_partition(self):
        base = TensorFeedConfig(
            users=400,
            steps=4,
            candidates=8,
            merged_candidates=24,
            batch_users=50,
            catalog_items=2_000,
            device="cpu",
        )
        small = run_tensor_feed(base, LOCAL_STATIC)
        large = run_tensor_feed(replace(base, batch_users=400), LOCAL_STATIC)
        self.assertEqual(
            small["candidate_graph"]["stage_attribution"],
            large["candidate_graph"]["stage_attribution"],
        )
        for metric in (
            "stay_per_exposure",
            "long_view_rate",
            "lt_value_per_user",
            "fine_oracle_regret_per_exposure",
        ):
            self.assertAlmostEqual(
                small["metrics"][metric], large["metrics"][metric], places=6
            )

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
        ablations = campaign_candidate_sets("local_ablation_v1")
        self.assertEqual(len(ablations), 32)
        self.assertNotIn(
            13,
            ablations[
                "basic__realtime__local_context__category_hash"
                "__without_poi_indicator"
            ],
        )
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

    def test_candidate_graph_respects_sparse_search_and_retarget_routes(self):
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
        state["search_ttl"] = torch.tensor([2, 0, 2, 0])
        state["search_strength"] = torch.ones(config.users)
        candidates = _candidate_batch(
            config, generator, torch.device("cpu"), state, catalog, step=0
        )
        self.assertEqual(candidates["item_ids"].shape, (4, 5))
        self.assertTrue(torch.all(candidates["route_valid_counts"][:, 7] > 0))
        self.assertEqual(candidates["route_valid_counts"][1, 6], 0)
        self.assertEqual(candidates["route_valid_counts"][3, 6], 0)
        self.assertTrue(candidates["audit_oracle_in_recall"].dtype == torch.bool)

    def test_gpu_candidate_graph_exercises_real_stage_attrition(self):
        report = run_tensor_feed(
            TensorFeedConfig(
                users=400,
                steps=3,
                candidates=10,
                merged_candidates=30,
                batch_users=100,
                catalog_items=2_000,
                trace_users=2,
                trace_requests_per_user=2,
                device="cpu",
            ),
            LOCAL_STATIC,
        )
        graph = report["candidate_graph"]
        self.assertEqual(
            sum(graph["stage_attribution"].values()), graph["requests"]
        )
        self.assertGreater(graph["mean_unique_recalled"], 10)
        self.assertGreater(graph["stage_attribution"]["recall_miss"], 0)
        self.assertLess(report["metrics"]["coarse_pass_fraction"], 1.0)
        trace = report["request_candidate_trace"]
        self.assertEqual(trace["schema"], "gpu-request-candidate-trace-v1")
        self.assertEqual(trace["candidate_rows"], trace["requests"] * 30)
        for request_id in {row["request_id"] for row in trace["rows"]}:
            request_rows = [
                row for row in trace["rows"] if row["request_id"] == request_id
            ]
            self.assertEqual(sum(row["exposed"] for row in request_rows), 1)
            self.assertEqual(
                sum(bool(row["mature_labels"]) for row in request_rows), 1
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
        self.assertAlmostEqual(
            report["metrics"]["coarse_pass_fraction"],
            5 / config.merged_candidates,
        )
        self.assertGreaterEqual(report["metrics"]["coarse_feed_oracle_recall"], 0.0)
        self.assertLessEqual(report["metrics"]["coarse_feed_oracle_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
