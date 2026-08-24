"""Independent check for the same-user counterfactual estimator."""

import unittest

import torch

from fid_lab.feed_loop.scale.experiment.trigger import (
    combine_tensor_counterfactual_ab,
)
from fid_lab.feed_loop.scale.graph.reporting import CELL_METRICS


class PairedEstimatorTest(unittest.TestCase):
    def test_common_user_noise_cancels_from_standard_error(self):
        users = 10_000
        common = torch.linspace(-5.0, 5.0, users)[:, None]
        control = common.expand(users, len(CELL_METRICS)).clone()
        treatment = control + 0.02
        report = combine_tensor_counterfactual_ab(
            {"_paired_user_metrics": control},
            {"_paired_user_metrics": treatment},
        )
        metric = report[CELL_METRICS[0]]
        self.assertAlmostEqual(
            metric["treatment_mean"] - metric["control_mean"], 0.02, places=6
        )
        self.assertLess(metric["standard_error"], 1e-7)
        self.assertEqual(metric["estimator"], "same_user_paired_difference")


if __name__ == "__main__":
    unittest.main()
