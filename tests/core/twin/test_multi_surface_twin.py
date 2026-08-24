from __future__ import annotations

import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import torch

from fid_lab.simulation.twin.contracts import (
    ITEM_KINDS,
    SURFACE_CONTRACTS,
    ItemKind,
    Surface,
    TwinConfig,
    TwinPolicy,
)
from fid_lab.simulation.twin.experimentation.campaign import run_launch_campaign
from fid_lab.simulation.twin.experimentation.campaign import launch_decision
from fid_lab.simulation.twin.experimentation.experiment import run_twin_experiment
from fid_lab.simulation.twin.kernel import DigitalTwinKernel
from fid_lab.simulation.twin.ledger import candidate_history_signals
from fid_lab.simulation.twin.profiles import load_profile
from fid_lab.simulation.twin.exchange import ObservableResponse, ServedSlate, TASKS
from fid_lab.simulation.twin.platform.updates import apply_response_events
from fid_lab.simulation.twin.serving.surfaces import (
    CANDIDATE_FEATURES,
    build_slate,
)


def small_config():
    return TwinConfig(
        users=128,
        catalog_items=900,
        creators=90,
        topics=8,
        countries=4,
        preperiod_steps=2,
        measurement_steps=3,
        steps_per_day=2,
        history_length=8,
        route_candidates=4,
        routes=6,
        coarse_keep=12,
        fine_keep=5,
        audit_users=16,
        batch_users=64,
        device="cpu",
    )


def baseline_policy():
    return TwinPolicy(name="shared_rules_v1")


def assert_tensor_dataclass_equal(testcase, left, right):
    for field in fields(left):
        testcase.assertTrue(torch.equal(
            getattr(left, field.name), getattr(right, field.name),
        ), field.name)


def assert_platform_user_equal(testcase, left, right):
    for field in fields(left):
        if field.name == "ledger":
            assert_tensor_dataclass_equal(testcase, left.ledger, right.ledger)
            continue
        testcase.assertTrue(torch.equal(
            getattr(left, field.name), getattr(right, field.name),
        ), field.name)


class MultiSurfaceTwinTest(unittest.TestCase):
    def test_contract_covers_business_surfaces_and_content_shapes(self):
        self.assertEqual(set(SURFACE_CONTRACTS), set(Surface))
        self.assertEqual(len(ITEM_KINDS), len(ItemKind))
        for required in (
            ItemKind.SHORT_VIDEO, ItemKind.PHOTO, ItemKind.ARTICLE,
            ItemKind.CARD, ItemKind.LIVE_ROOM, ItemKind.PRODUCT,
            ItemKind.POI, ItemKind.AD,
        ):
            self.assertIn(required.name.lower(), ITEM_KINDS)
        capacity = TwinConfig().manifest()["capacity"]
        self.assertEqual(capacity["surfaces"], 6)
        self.assertEqual(capacity["item_kinds"], 9)
        self.assertEqual(capacity["trajectory_steps"], 40)
        self.assertEqual(capacity["candidate_rows_per_request"], 96)
        gpu = load_profile("gpu", "cpu")
        self.assertEqual(gpu.users, 1_000_000)
        self.assertEqual(gpu.catalog_items, 2_000_000)
        self.assertEqual(gpu.batch_users, 250_000)
        self.assertEqual(gpu.history_length, 64)

    def test_snapshot_fork_is_deep_and_preperiod_is_single_materialization(self):
        kernel = DigitalTwinKernel(small_config())
        preperiod = kernel.preperiod(baseline_policy())
        left = preperiod.snapshot.fork()
        right = preperiod.snapshot.fork()
        before = right.users[0].satisfaction_estimate.clone()
        left.users[0].satisfaction_estimate.zero_()
        self.assertTrue(torch.equal(
            right.users[0].satisfaction_estimate, before
        ))
        self.assertEqual(preperiod.snapshot.step, 2)
        self.assertEqual(len(preperiod.snapshot.preperiod_user_metrics), 2)
        left.context.global_topic_heat.zero_()
        self.assertGreater(float(right.context.global_topic_heat.sum()), 0.0)

    def test_platform_state_contains_estimates_not_hidden_truth(self):
        snapshot = DigitalTwinKernel(small_config()).initialize()
        observed = snapshot.users[0]
        hidden = snapshot.latent_users[0]
        self.assertFalse(torch.equal(
            observed.satisfaction_estimate, hidden.satisfaction
        ))
        self.assertFalse(torch.equal(
            observed.commerce_intent_estimate, hidden.commerce_intent
        ))
        future = ~observed.registered
        self.assertTrue((observed.signup_step[future] == -1).all())
        self.assertTrue((hidden.signup_step[future] > 0).all())
        self.assertFalse(hasattr(observed, "retained"))
        self.assertFalse(hasattr(observed, "long_interest"))

    def test_platform_initial_state_is_invariant_to_hidden_world_seed(self):
        config = small_config()
        factual = DigitalTwinKernel(config).initialize()
        counterfactual = DigitalTwinKernel(replace(
            config, environment_seed=config.environment_seed + 1009
        )).initialize()
        for left_users, right_users in zip(
            factual.users, counterfactual.users, strict=True
        ):
            assert_platform_user_equal(self, left_users, right_users)
        for field in fields(factual.catalog):
            self.assertTrue(torch.equal(
                getattr(factual.catalog, field.name),
                getattr(counterfactual.catalog, field.name),
            ), field.name)
        self.assertFalse(torch.equal(
            factual.latent_users[0].long_interest,
            counterfactual.latent_users[0].long_interest,
        ))
        self.assertFalse(torch.equal(
            factual.latent_catalog.semantic_embedding,
            counterfactual.latent_catalog.semantic_embedding,
        ))

    def test_no_observable_event_cannot_change_interest_state(self):
        snapshot = DigitalTwinKernel(small_config()).initialize()
        users = snapshot.users[0]
        before_short = users.short_interest.clone()
        before_observed = users.observed_interest.clone()
        count = len(users.user_id)
        response = ObservableResponse(
            task=torch.zeros(count, len(TASKS), dtype=torch.bool),
            task_mask=torch.zeros(count, len(TASKS), dtype=torch.bool),
            stay_seconds=torch.zeros(count),
            selected_item=torch.zeros(count, dtype=torch.long),
            active=torch.zeros(count, dtype=torch.bool),
        )
        apply_response_events(
            users, snapshot.catalog, response,
            torch.full((count,), int(Surface.FEED), dtype=torch.long),
        )
        self.assertTrue(torch.equal(users.short_interest, before_short))
        self.assertTrue(torch.equal(users.observed_interest, before_observed))

    def test_platform_ranking_is_invariant_to_hidden_world_intervention(self):
        kernel = DigitalTwinKernel(small_config())
        snapshot = kernel.initialize()
        users = snapshot.users[0]
        surface = torch.full_like(users.user_id, int(Surface.FEED))
        before = build_slate(
            kernel.config, baseline_policy(), users, snapshot.catalog,
            snapshot.context, surface, 0,
        )
        snapshot.latent_users[0].long_interest = (
            snapshot.latent_users[0].long_interest.roll(3, dims=1)
        )
        snapshot.latent_users[0].satisfaction.zero_()
        snapshot.latent_catalog.true_quality.zero_()
        snapshot.latent_catalog.true_risk.fill_(1.0)
        after = build_slate(
            kernel.config, baseline_policy(), users, snapshot.catalog,
            snapshot.context, surface, 0,
        )
        for name in (
            "item_ids", "recall_score", "coarse_score", "fine_score",
            "eligible", "exposed_item_ids", "feature_values",
        ):
            self.assertTrue(torch.equal(
                getattr(before, name), getattr(after, name)
            ), name)

    def test_platform_modules_cannot_import_hidden_environment(self):
        twin = Path(__file__).parents[3] / "fid_lab/simulation/twin"
        for package in ("platform", "serving", "training"):
            for path in (twin / package).rglob("*.py"):
                source = path.read_text()
                self.assertNotIn("environment.", source, str(path))
                self.assertNotIn("LatentUserState", source, str(path))
                self.assertNotIn("LatentCatalogState", source, str(path))
                self.assertNotIn("true_quality", source, str(path))
                self.assertNotIn("true_risk", source, str(path))
        self.assertFalse(any(
            "latent" in name or name.startswith("true_")
            for name in CANDIDATE_FEATURES
        ))

    def test_held_out_environment_changes_response_not_platform_slate(self):
        config = small_config()
        kernel = DigitalTwinKernel(config)
        snapshot = kernel.initialize()
        users = snapshot.users[0]
        hidden = snapshot.latent_users[0]
        surface = torch.full_like(users.user_id, int(Surface.FEED))
        candidates = build_slate(
            config, baseline_policy(), users, snapshot.catalog,
            snapshot.context, surface, 0,
        )
        slate = ServedSlate(candidates.exposed_item_ids)
        factual = kernel.environment.respond(
            users, hidden, snapshot.catalog, snapshot.latent_catalog,
            snapshot.context, slate, surface, 0,
        )
        held_out = DigitalTwinKernel(replace(
            config, environment_seed=config.environment_seed + 1009
        ))
        counterfactual = held_out.environment.respond(
            users, hidden, snapshot.catalog, snapshot.latent_catalog,
            snapshot.context, slate, surface, 0,
        )
        self.assertTrue(torch.equal(
            candidates.exposed_item_ids, slate.exposed_item_ids
        ))
        self.assertFalse(torch.equal(factual.task, counterfactual.task))

    def test_exposure_ledger_detects_cross_request_item_author_and_cluster(self):
        kernel = DigitalTwinKernel(small_config())
        snapshot = kernel.initialize()
        users = snapshot.users[0]
        candidate = torch.arange(4)[None, :].expand(len(users.user_id), -1)
        users.ledger.item[:, 0] = candidate[:, 0]
        users.ledger.author[:, 0] = snapshot.catalog.author[candidate[:, 0]]
        users.ledger.cluster[:, 0] = snapshot.catalog.cluster[candidate[:, 0]]
        users.ledger.topic[:, 0] = snapshot.catalog.topic[candidate[:, 0]]
        users.ledger.kind[:, 0] = snapshot.catalog.kind[candidate[:, 0]]
        users.ledger.step[:, 0] = 0
        signals = candidate_history_signals(
            users.ledger, snapshot.catalog, candidate, 1
        )
        self.assertTrue(signals["repeated_item"][:, 0].all())
        self.assertTrue((signals["author_fatigue"][:, 0] > 0).all())
        self.assertTrue((signals["cluster_fatigue"][:, 0] > 0).all())

    def test_aa_branches_have_identical_potential_outcomes(self):
        kernel = DigitalTwinKernel(small_config())
        policy = baseline_policy()
        preperiod = kernel.preperiod(policy)
        left = kernel.arm(preperiod.snapshot, policy)
        right = kernel.arm(preperiod.snapshot, policy)
        for left_values, right_values in zip(
            left.user_metrics, right.user_metrics, strict=True
        ):
            self.assertTrue(torch.equal(left_values, right_values))

    def test_candidate_microbatch_preserves_world_outcomes(self):
        baseline = small_config()
        chunked = replace(baseline, serve_chunk_users=17)
        full = replace(baseline, serve_chunk_users=baseline.batch_users)
        chunked_kernel = DigitalTwinKernel(chunked)
        chunked_run = chunked_kernel.run(
            chunked_kernel.initialize(), baseline_policy(), 3
        )
        full_kernel = DigitalTwinKernel(full)
        full_run = full_kernel.run(
            full_kernel.initialize(), baseline_policy(), 3
        )
        for left, right in zip(
            chunked_run.user_metrics, full_run.user_metrics, strict=True
        ):
            self.assertTrue(torch.equal(left, right))

    def test_experiment_closes_request_trace_and_reports_all_surfaces(self):
        experiment = run_twin_experiment(small_config())
        report = experiment.report
        self.assertEqual(report["preperiod"]["materializations"], 1)
        self.assertFalse(report["preperiod"]["recomputed_per_arm"])
        for arm in ("control", "treatment"):
            self.assertTrue(all(report["trace"]["gates"][arm].values()))
        summary = report["control_summary"]
        for surface in Surface:
            self.assertIn(f"{surface.name.lower()}_request_share", summary)
        self.assertIn("synthetic_lt_measurement", report["cuped_ab"])
        negative_rate = report["cuped_ab"]["negative_rate_per_request"]
        self.assertGreaterEqual(negative_rate["control_mean"], 0.0)
        self.assertLessEqual(negative_rate["control_mean"], 1.0)
        self.assertIn("experiment_cells", report["sample_evolution"])
        self.assertIn("country", report["sample_evolution"]["slices"])
        self.assertIn("cold_start", report["sample_evolution"]["slices"])
        self.assertIn("synthetic_lt_delta_vs_full_rollout", report[
            "ecosystem_interference"
        ])
        self.assertGreater(
            report["sample_evolution"]["experiment_cells"]["control"][
                "registered_rate"
            ], 0.0,
        )

    def test_launch_negative_guardrail_uses_rate_not_window_count(self):
        def metric(control, difference, interval):
            return {
                "control_mean": control,
                "difference": difference,
                "confidence_interval": list(interval),
            }
        experiment = SimpleNamespace(report={
            "cuped_ab": {
                "synthetic_lt_measurement": metric(3.0, 0.1, (0.05, 0.15)),
                "stay_seconds": metric(30.0, 0.2, (-0.1, 0.5)),
                "negative": metric(0.2, 0.02, (-0.01, 0.05)),
                "negative_rate_per_request": metric(
                    0.01, 0.0002, (-0.0004, 0.0015)
                ),
                "requests": metric(12.0, 0.0, (-0.02, 0.02)),
            },
            "trace": {"gates": {"control": {"closed": True}}},
        })
        decision, gates = launch_decision(experiment)
        self.assertEqual(decision, "pass")
        self.assertTrue(gates["negative_guardrail"])

    def test_continuous_campaign_preserves_last_accepted_control(self):
        report = run_launch_campaign(small_config())
        active = "shared_rules_v1"
        for launch in report["launches"]:
            self.assertEqual(launch["control"], active)
            if launch["decision"] == "pass":
                active = launch["candidate"]
            self.assertEqual(launch["active_after"], active)
        self.assertEqual(
            report["stage_counts"],
            {"retrieval": 2, "coarse": 2, "fine": 2, "mix": 2},
        )


if __name__ == "__main__":
    unittest.main()
