from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from fid_lab.poi_distribution.models.architectures import build_ranker
from fid_lab.poi_distribution.contracts import PoiDistributionTrainingConfig
from fid_lab.poi_distribution.release import build_poi_distribution_release
from fid_lab.poi_distribution.models.training import (
    PoiRankerBundle,
    load_bundle,
    save_bundle,
    task_probabilities,
)


class PoiDistributionV4Test(unittest.TestCase):
    def test_entire_space_probabilities_preserve_the_cascade(self):
        logits = {
            task: torch.randn(128)
            for task in (
                "anchor_click", "poi_detail", "poi_favorite", "conversion",
                "negative_feedback", "stay_norm",
            )
        }
        probability = task_probabilities(logits)
        self.assertTrue(torch.all(
            probability["poi_detail"] <= probability["anchor_click"]
        ))
        self.assertTrue(torch.all(
            probability["conversion"] <= probability["poi_detail"]
        ))
        self.assertTrue(torch.all(
            probability["poi_favorite"] <= probability["poi_detail"]
        ))

    def test_artifact_replays_exact_scores(self):
        config = PoiDistributionTrainingConfig(
            epochs=1, batch_size=32, device="cpu"
        )
        model = build_ranker("linear", config.feature_dim)
        bundle = PoiRankerBundle(
            "linear", model, torch.zeros(28), torch.ones(28),
            {task: 0.0 for task in (
                "anchor_click", "poi_detail", "poi_favorite", "conversion",
                "negative_feedback", "stay_norm",
            )},
            {"validation": {}},
        )
        features = torch.rand(64, 28)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "linear.pt"
            save_bundle(bundle, path, config)
            loaded = load_bundle(path)
            before = bundle.score(features)
            after = loaded.score(features)
        self.assertTrue(torch.equal(before, after))

    def test_release_binds_passed_coarse_fine_and_end_to_end(self):
        root = Path(__file__).resolve().parents[2]
        release = build_poi_distribution_release(
            root,
            "reports/training/2026-08-24-poi-distribution-v4-training.json",
            "reports/launches/2026-08-24-poi-distribution-v4-coarse-1m.json",
            "reports/launches/2026-08-24-poi-distribution-v4-fine-mix-200k.json",
            "reports/launches/2026-08-24-poi-distribution-v4-e2e-500k.json",
            "artifacts/models/poi-distribution-v4",
            "reports/training/2026-08-24-shared-retrieval-v4-aligned-training.json",
            "reports/launches/2026-08-24-shared-retrieval-v4-aligned-paired-500k.json",
            "artifacts/models/shared-retrieval-v4-aligned",
        )
        self.assertEqual(
            release["active_bundle"]["coarse_model"], "linear"
        )
        self.assertEqual(release["active_bundle"]["fine_model"], "linear")
        self.assertEqual(
            release["active_bundle"]["retrieval_policy"],
            "shared_two_tower_ann_graph_geo_fresh_tail_popular_search_retarget",
        )
        self.assertEqual(
            release["production_readiness"],
            "simulator_only_external_local_validation_required",
        )


if __name__ == "__main__":
    unittest.main()
