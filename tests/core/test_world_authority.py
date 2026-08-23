from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fid_lab.feed_loop.world_model.release import (
    build_composite_world_review,
    build_world_release,
)


class CompositeWorldAuthorityTest(unittest.TestCase):
    def test_task_kernels_promote_without_claiming_production_authority(self):
        root = Path(__file__).resolve().parents[2]
        review = build_composite_world_review(root)
        self.assertEqual(
            review["schema"], "composite-recommendation-world-review-v1"
        )
        self.assertEqual(
            review["decision"], "promote_feed_and_local_kernels"
        )
        self.assertEqual(
            review["components"]["feed_behavior"]["status"],
            "eligible_simulator_authority",
        )
        self.assertEqual(
            review["components"]["unified_neural_scm"]["status"],
            "hold_research_challenger",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(review))
            release = build_world_release(path)
        self.assertEqual(release["production_readiness"], "simulator_only")
        self.assertEqual(
            release["active_components"]["local_response"]["authority"],
            "synthetic_neural_v4",
        )
