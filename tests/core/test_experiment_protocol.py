from __future__ import annotations

import unittest

from fid_lab.launches.experiment_protocol import (
    ExperimentPhase,
    ExperimentPlan,
    payload_fingerprint,
    phase_decision,
)


class ExperimentProtocolTest(unittest.TestCase):
    def _plan(self, phase=ExperimentPhase.SCREEN, **changes):
        values = {
            "launch_id": "L-TEST-001",
            "phase": phase,
            "hypothesis": "one isolated change improves LT",
            "isolated_change": "risk threshold 0.75",
            "primary_metric": "lt_value_per_user",
            "mde_absolute": 0.01,
            "alpha": 0.05,
            "power": 0.80,
            "pilot_total_users": 100_000,
            "pilot_primary_standard_error": 0.01,
            "users_per_salt": 100_000,
            "salts": (11, 23, 47),
            "control_fingerprint": payload_fingerprint({"name": "control"}),
            "treatment_fingerprint": payload_fingerprint({"name": "treatment"}),
            "scenario_fingerprint": payload_fingerprint({"world": "v5"}),
            "predecessor_report": "reports/launches/smoke.json",
            "predecessor_report_sha256": "a" * 64,
            "registered_before_evidence": True,
        }
        values.update(changes)
        return ExperimentPlan(**values)

    def test_screen_has_one_fixed_scale_and_cannot_pass(self):
        plan = self._plan()
        self.assertEqual(plan.planned_total_users, 300_000)
        self.assertEqual(
            phase_decision(plan, "pass", plan.salts), "advance_to_powered"
        )

    def test_single_salt_is_partial_evidence(self):
        plan = self._plan()
        self.assertEqual(
            phase_decision(plan, "pass", (plan.salts[0],)),
            "partial_evidence",
        )

    def test_powered_plan_must_cover_preregistered_mde(self):
        with self.assertRaisesRegex(ValueError, "below its pre-registered MDE"):
            self._plan(
                ExperimentPhase.POWERED_AB,
                pilot_primary_standard_error=0.03,
            )

    def test_post_hoc_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Post-hoc|post-hoc"):
            self._plan(registered_before_evidence=False)

    def test_scale_benchmark_cannot_make_a_business_decision(self):
        plan = self._plan(
            ExperimentPhase.SCALE_BENCHMARK,
            users_per_salt=1_000_000,
            salts=(11,),
            predecessor_report=None,
            predecessor_report_sha256=None,
        )
        self.assertEqual(
            phase_decision(plan, "pass", plan.salts), "benchmark_pass"
        )
        self.assertEqual(
            phase_decision(plan, "hold_or_reject", plan.salts),
            "benchmark_fail",
        )

    def test_runtime_must_match_artifacts_scenario_scale_and_salt(self):
        plan = self._plan()
        plan.validate_run(
            {"name": "control"}, {"name": "treatment"}, {"world": "v5"},
            100_000, 23,
        )
        with self.assertRaisesRegex(ValueError, "runtime users"):
            plan.validate_run(
                {"name": "control"}, {"name": "treatment"}, {"world": "v5"},
                200_000, 23,
            )


if __name__ == "__main__":
    unittest.main()
