"""Creator-cluster estimators must not treat repeated requests as users."""

import unittest

import torch

from fid_lab.launches.statistics import (
    cluster_paired_metric,
    cluster_randomized_metric,
)


class ClusterStatisticsTest(unittest.TestCase):
    def test_cluster_paired_effect_uses_creator_as_unit(self):
        creators, requests = 2_000, 8
        creator = torch.arange(creators).repeat(requests)
        common = torch.sin(creator.float())
        report = cluster_paired_metric(common, common + 0.03, creator)
        self.assertAlmostEqual(report["absolute_effect"], 0.03, places=6)
        self.assertEqual(report["clusters"], creators)

    def test_randomized_assignment_keeps_all_creator_requests_together(self):
        creators, requests = 10_000, 4
        creator = torch.arange(creators).repeat(requests)
        common = torch.cos(creator.float())
        report = cluster_randomized_metric(common, common + 0.02, creator)
        self.assertEqual(
            report["control_creators"] + report["treatment_creators"], creators
        )
        self.assertEqual(report["estimator"], "creator_cluster_randomized_ab")

    def test_sparse_creator_ids_do_not_create_fake_zero_clusters(self):
        creator = torch.tensor([101, 101, 9001, 9001])
        control = torch.tensor([1.0, 1.0, 3.0, 3.0])
        report = cluster_paired_metric(control, control + 0.5, creator)
        self.assertEqual(report["clusters"], 2)
        self.assertAlmostEqual(report["absolute_effect"], 0.5)


if __name__ == "__main__":
    unittest.main()
