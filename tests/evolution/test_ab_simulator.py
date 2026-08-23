from __future__ import annotations

import unittest

from fid_lab.evolution.evaluation.ab_simulator import (
    SCENARIOS,
    run_scenario_suite,
    simulate_experiment,
)
from fid_lab.simulation.ab import launch_decision


class OnlineExperimentSimulatorTest(unittest.TestCase):
    def test_model_experiment_recovers_injected_itt(self) -> None:
        report = simulate_experiment(SCENARIOS["model"], users=200_000)
        self.assertTrue(report["truth_covered"])
        self.assertGreater(report["metrics"]["watch_minutes"]["absolute_lift"], 0.0)
        self.assertLess(report["metrics"]["negative_feedback"]["absolute_lift"], 0.0)

    def test_product_model_and_strategy_suite_reports_guardrails(self) -> None:
        report = run_scenario_suite(users=200_000)
        self.assertTrue(report["all_truth_covered"])
        self.assertEqual(set(report["reports"]), {"product", "model", "strategy"})

    def test_unexchanged_primary_regression_does_not_override_lt_uncertainty(self) -> None:
        metrics = {
            "negative_feedback": {"absolute_lift": 0.0, "p_value": 1.0},
            "quality_long_view_rate": {"relative_lift": 0.01, "p_value": 0.2},
            "stay_per_exposure": {"absolute_lift": -0.1, "p_value": 0.01},
            "lt_value": {
                "absolute_lift": -0.02,
                "relative_lift": -0.02,
                "confidence_interval": (-0.08, 0.04),
                "p_value": 0.3,
            },
        }
        self.assertEqual(launch_decision(metrics), "hold_unified_lt_uncertain")


if __name__ == "__main__":
    unittest.main()
