from __future__ import annotations

import unittest

from fid_lab.evolution.evaluation.ab_simulator import (
    SCENARIOS,
    run_scenario_suite,
    simulate_experiment,
)


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


if __name__ == "__main__":
    unittest.main()
