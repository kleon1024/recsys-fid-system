import unittest

from fid_lab.feed_loop.scale.queue_value_cli import apply_exchange_rate_gate
from fid_lab.value import (
    DEFAULT_LT_CONFIG,
    BusinessValueSignals,
    BusinessValueTree,
    LTMetricContainer,
    LTMetricVector,
)


class ValueBoundaryTest(unittest.TestCase):
    def test_local_tree_uses_deepest_consumption_action(self) -> None:
        result = BusinessValueTree().evaluate(
            BusinessValueSignals(anchor_click=1, poi_detail=1, poi_favorite=1)
        )
        self.assertEqual(result.local_consumption, 3.0)

    def test_lt_accepts_platform_metrics_only(self) -> None:
        metrics = LTMetricVector(
            stay_minutes=2.0,
            active_days=1.0,
            accepted_commercialization_value=3.0,
        )
        result = LTMetricContainer().evaluate(metrics)
        expected = (
            2.0 * DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value
            + DEFAULT_LT_CONFIG.rates["active_day"].unit_value
            + 3.0
            * DEFAULT_LT_CONFIG.rates["accepted_commercialization_unit"].unit_value
        )
        self.assertEqual(result.total, expected)

    def test_local_tree_score_is_not_an_lt_input(self) -> None:
        local = BusinessValueTree().evaluate(
            BusinessValueSignals(
                anchor_click=1,
                poi_detail=1,
                closed_loop_payment=1,
                contribution_margin=2.0,
            )
        )
        lt = LTMetricContainer().evaluate(LTMetricVector())
        self.assertGreater(local.local_transaction, 0.0)
        self.assertEqual(lt.total, 0.0)

    def test_synthetic_commercialization_rate_cannot_pass_launch(self) -> None:
        report = {
            "aggregate": [
                {
                    "decision": "pass_lt_value",
                    "metrics": {
                        "accepted_platform_commercialization_per_exposure": {
                            "pooled_absolute_lift": 0.01,
                        },
                    },
                    "lt_exchange_sensitivity": {
                        "0": {
                            "pooled_absolute_lift": -0.001,
                            "pooled_p_value": 0.5,
                            "known_mean_absolute_effect": -0.001,
                        },
                        "1": {
                            "pooled_absolute_lift": 0.01,
                            "pooled_p_value": 0.01,
                            "known_mean_absolute_effect": 0.01,
                        },
                    },
                }
            ]
        }
        gated = apply_exchange_rate_gate(report)
        self.assertEqual(
            gated["aggregate"][0]["decision"],
            "hold_exchange_rate_unvalidated",
        )


if __name__ == "__main__":
    unittest.main()
