import unittest

import numpy as np

from fid_lab.evolution.evaluation.metrics import grouped_auc


class GroupedAUCTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
