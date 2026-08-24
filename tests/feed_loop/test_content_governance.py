from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from fid_lab.feed_loop.governance import (
    ContentGovernanceConfig,
    evaluate_governance_launch,
    govern_scores,
)
from fid_lab.feed_loop.scale.tensor_runtime.behavior.external import (
    ExternalSequenceMixtureWorld,
)
from fid_lab.feed_loop.scale.tensor_runtime.state_transition import (
    sample_terminal_retention,
)
from fid_lab.feed_loop.scale.graph.reporting import CELL_METRICS
from fid_lab.feed_loop.serving.aggregate import aggregate_governance_launches
from fid_lab.launches.experiment_protocol import (
    ExperimentPhase,
    ExperimentPlan,
    payload_fingerprint,
)


class ContentGovernanceTest(unittest.TestCase):
    def _inputs(self):
        candidates = {
            "predicted_integrity_risk": torch.tensor([[0.1, 0.9, 0.2]]),
            "is_poi": torch.tensor([[False, False, False]]),
            "duplicate_cluster": torch.tensor([[7, 8, 9]]),
            "author": torch.tensor([[3, 4, 5]]),
            "creator_need": torch.zeros(1, 3),
        }
        state = {
            "poi_served": torch.zeros(1, dtype=torch.long),
            "last_poi_step": torch.full((1,), -10_000, dtype=torch.long),
            "last_duplicate_cluster": torch.tensor([7]),
            "last_author": torch.tensor([3]),
        }
        return candidates, state

    def test_filter_respects_upstream_candidate_mask(self):
        candidates, state = self._inputs()
        scores = torch.tensor([[3.0, 2.0, 1.0]])
        upstream = torch.tensor([[False, True, True]])
        governed, diagnostics = govern_scores(
            scores, candidates, state, 0, ContentGovernanceConfig(), upstream
        )
        self.assertEqual(int(governed.argmax(dim=1)[0]), 2)
        self.assertFalse(diagnostics["governance_eligible"][0, 0])
        self.assertFalse(diagnostics["governance_eligible"][0, 1])

    def test_empty_safe_set_falls_back_only_within_upstream_candidates(self):
        candidates, state = self._inputs()
        candidates["predicted_integrity_risk"][:] = torch.tensor(
            [[0.99, 0.98, 0.97]]
        )
        upstream = torch.tensor([[False, True, True]])
        governed, diagnostics = govern_scores(
            torch.zeros(1, 3), candidates, state, 0,
            ContentGovernanceConfig(), upstream,
        )
        self.assertEqual(int(governed.argmax(dim=1)[0]), 2)
        self.assertTrue(diagnostics["governance_fallback"][0])

    def test_governance_review_uses_observable_metrics(self):
        names = {
            "predicted_integrity_risk_per_exposure": -0.02,
            "near_duplicate_rate": -0.01,
            "lt_value_per_user": 0.002,
            "stay_per_exposure": 0.001,
            "quality_long_view_rate": 0.001,
            "negative_rate": -0.001,
        }
        metrics = {
            name: {
                "control_mean": 0.2,
                "treatment_mean": 0.2 + effect,
                "confidence_interval": (effect - 0.0001, effect + 0.0001),
                "standard_error": 0.00005,
            }
            for name, effect in names.items()
        }
        review = evaluate_governance_launch(
            metrics, metrics,
            ContentGovernanceConfig(max_poi_per_session=100, min_poi_gap=0),
            trajectory_steps=8, population=10_000,
        )
        self.assertEqual(review["decision"], "pass")
        self.assertNotIn("oracle", " ".join(review["online_gates"]))

    def test_hidden_repetition_response_has_expected_behavior_direction(self):
        probability = torch.full((2, 7), 0.5)
        stay = torch.full((2,), 0.5)
        state = {
            "last_duplicate_cluster": torch.tensor([4, 9]),
            "last_author": torch.tensor([2, 8]),
            "hidden_novelty": torch.tensor([0.8, 0.8]),
            "hidden_patience": torch.tensor([0.2, 0.2]),
        }
        candidates = {
            "duplicate_cluster": torch.tensor([[4], [10]]),
            "author": torch.tensor([[2], [7]]),
        }
        adjusted, adjusted_stay = (
            ExternalSequenceMixtureWorld._apply_repetition_response(
                probability, stay, state, candidates
            )
        )
        self.assertLess(adjusted[0, 1], adjusted[1, 1])
        self.assertGreater(adjusted[0, 6], adjusted[1, 6])
        self.assertLess(adjusted_stay[0], adjusted_stay[1])

    def test_terminal_retention_has_equal_opportunity_and_monotone_state(self):
        users = 10_000
        common = {
            "user_ids": torch.arange(users),
            "historical_activity": torch.full((users,), 10.0),
            "hidden_patience": torch.full((users,), 0.5),
        }
        low = {
            **common,
            "hidden_satisfaction": torch.full((users,), -0.5),
            "hidden_fatigue": torch.full((users,), 0.8),
        }
        high = {
            **common,
            "hidden_satisfaction": torch.full((users,), 0.5),
            "hidden_fatigue": torch.full((users,), 0.2),
        }
        config = SimpleNamespace(steps=8, seed=20260823)
        low_label = sample_terminal_retention(config, low)
        high_label = sample_terminal_retention(config, high)
        self.assertTrue(torch.all(high_label >= low_label))
        self.assertGreater(high_label.float().mean(), low_label.float().mean())

    def test_governance_aggregate_requires_replicated_directions(self):
        def metrics():
            effects = {
                "lt_value_per_user": 0.01,
                "stay_per_exposure": 0.01,
                "quality_long_view_rate": 0.001,
                "negative_rate": -0.001,
                "predicted_integrity_risk_per_exposure": -0.01,
                "near_duplicate_rate": -0.01,
            }
            return {
                name: {
                    "control_mean": 0.2,
                    "treatment_mean": 0.2 + effects.get(name, 0.001),
                    "standard_error": 0.0001,
                    "confidence_interval": (
                        effects.get(name, 0.001) - 0.000196,
                        effects.get(name, 0.001) + 0.000196,
                    ),
                }
                for name in CELL_METRICS
            }

        governance = ContentGovernanceConfig().manifest()
        control = {"name": "control"}
        treatment = {"name": "treatment", "content_governance": governance}
        world = {"authority": "v5"}
        scenario = {
            "config": {"steps": 8},
            "measurement_start_step": 0,
            "behavior_world": world,
        }
        plan = ExperimentPlan(
            launch_id="L-GOV-TEST", phase=ExperimentPhase.SCREEN,
            hypothesis="governance improves LT", isolated_change="governance",
            primary_metric="lt_value_per_user", mde_absolute=0.01,
            alpha=0.05, power=0.80, pilot_total_users=100_000,
            pilot_primary_standard_error=0.01, users_per_salt=100_000,
            salts=(11, 23, 47), control_fingerprint=payload_fingerprint(control),
            treatment_fingerprint=payload_fingerprint(treatment),
            scenario_fingerprint=payload_fingerprint(scenario),
            predecessor_report="reports/launches/smoke.json",
            predecessor_report_sha256="a" * 64,
            registered_before_evidence=True,
        )
        reports = [
            {
                "schema": "content-governance-launch-v1",
                "config": {
                    "experiment_salt": salt, "users": 100_000, "steps": 8,
                },
                "warmup_steps": 0,
                "control": control,
                "treatment": treatment,
                "behavior_world": world,
                "experiment_plan": plan.manifest(),
                "experiment_plan_fingerprint": plan.plan_fingerprint,
                "paired_shadow_replay": metrics(),
                "online_cuped_ab": metrics(),
            }
            for salt in (11, 23, 47)
        ]
        aggregate = aggregate_governance_launches(reports)
        self.assertEqual(aggregate["schema"], "content-governance-aggregate-v1")
        self.assertEqual(aggregate["decision"], "advance_to_powered")
        self.assertTrue(aggregate["online_gates"]["lt_direction_replicated"])


if __name__ == "__main__":
    unittest.main()
