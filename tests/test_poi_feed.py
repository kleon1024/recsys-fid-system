from __future__ import annotations

import unittest

import numpy as np

from fid_lab.poi_feed import (
    FeedAction,
    FeedImpression,
    FullPathConsistencyAuditor,
    PoiFeedJoiner,
    ViewerBehaviorEvent,
    ViewerFeatureOperator,
)


class PoiFeedPipelineTest(unittest.TestCase):
    def test_sequence_snapshot_is_point_in_time(self) -> None:
        operator = ViewerFeatureOperator(allowed_lateness_seconds=10)
        operator.ingest(ViewerBehaviorEvent("a", 1, 3, "view", 100, 101), 95)
        operator.ingest(ViewerBehaviorEvent("b", 1, 5, "favorite", 200, 201), 195)
        snapshot = operator.snapshot(1, 150)
        self.assertEqual(snapshot.category_sequence, (3,))
        self.assertEqual(snapshot.count_1h, 1)

    def test_joiner_extracts_only_anchored_main_feed_samples(self) -> None:
        operator = ViewerFeatureOperator()
        common = {
            "viewer_id": 1,
            "author_id": 2,
            "category_id": 3,
            "event_time": 100,
            "base_features": (0.0,) * 10,
            "media_version": "m",
            "feature_version": "f",
            "model_version": "r",
            "index_version": "i",
        }
        impressions = [
            FeedImpression("anchor", video_id=4, poi_id=5, **common),
            FeedImpression("plain", video_id=6, poi_id=None, **common),
        ]
        actions = [FeedAction("x", "anchor", "anchor_click", 120, 121)]
        report = PoiFeedJoiner().build(impressions, actions, operator, 700_000)
        self.assertEqual(report.main_impressions, 2)
        self.assertEqual(report.anchored_impressions, 1)
        self.assertEqual(len(report.examples), 1)
        self.assertEqual(report.examples[0].labels["anchor_click"], 1.0)

    def test_full_path_audit_checks_versions_and_cascade(self) -> None:
        operator = ViewerFeatureOperator()
        impression = FeedImpression(
            "anchor", 1, 2, 3, 4, 5, 100, (0.0,) * 10, "m", "f", "r", "i"
        )
        example = PoiFeedJoiner().build([impression], [], operator, 700_000).examples
        audit = FullPathConsistencyAuditor().audit(
            example,
            {"media": "m", "feature": "f", "model": "r", "index": "i"},
            np.zeros((1, 10)),
            np.zeros((1, 10)),
            {1},
            {1},
            {1},
        )
        self.assertTrue(audit.passed)


if __name__ == "__main__":
    unittest.main()
