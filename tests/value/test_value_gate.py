import unittest

from fid_lab.value import unified_lt_exchange_report, unified_lt_launch_decision


class UnifiedLTGateTest(unittest.TestCase):
    @staticmethod
    def metric(point: float, lower: float, upper: float) -> dict[str, object]:
        return {
            "absolute_lift": point,
            "confidence_interval": (lower, upper),
        }

    def test_positive_lt_with_nonnegative_lower_bound_passes(self) -> None:
        self.assertEqual(
            unified_lt_launch_decision(self.metric(0.03, 0.001, 0.06)),
            "pass_unified_lt_nonnegative",
        )

    def test_entirely_negative_lt_interval_rejects(self) -> None:
        self.assertEqual(
            unified_lt_launch_decision(self.metric(-0.03, -0.06, -0.001)),
            "reject_unified_lt_negative",
        )

    def test_interval_crossing_zero_holds(self) -> None:
        self.assertEqual(
            unified_lt_launch_decision(self.metric(0.01, -0.02, 0.04)),
            "hold_unified_lt_uncertain",
        )

    def test_hard_constraint_is_independent_of_lt(self) -> None:
        self.assertEqual(
            unified_lt_launch_decision(
                self.metric(0.03, 0.001, 0.06),
                hard_constraint_failure="reject_safety_constraint",
            ),
            "reject_safety_constraint",
        )

    def test_synthetic_exchange_rates_cannot_claim_production_readiness(self) -> None:
        metrics = {
            name: {
                "control_mean": 1.0,
                "treatment_mean": 1.1,
                "confidence_interval": (0.01, 0.19),
            }
            for name in (
                "lt_stay_per_user",
                "lt_active_days_per_user",
                "accepted_platform_commercialization_per_user",
                "lt_value_per_user",
            )
        }
        report = unified_lt_exchange_report(metrics)
        self.assertTrue(report["overall_nonnegative"])
        self.assertFalse(report["production_exchange_authority_accepted"])
        self.assertEqual(report["production_readiness"], "hold_synthetic_rates")


if __name__ == "__main__":
    unittest.main()
