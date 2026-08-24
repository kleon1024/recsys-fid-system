from __future__ import annotations

from pathlib import Path
import unittest

import torch

from fid_lab.poi_posting.world import PostingWorldConfig
from fid_lab.poi_posting.world.generator import (
    build_world,
    candidate_features,
    retrieve,
    rule_score,
    simulate_response,
)
from fid_lab.poi_posting.world.release import build_posting_release


class SupplyV4Test(unittest.TestCase):
    def test_creator_panel_and_mature_labels_match_the_logging_contract(self):
        config = PostingWorldConfig(
            requests=800,
            creators=100,
            cities=8,
            categories=4,
            items_per_cell=32,
            semantic_dim=8,
            train_epochs=1,
            world_version="creator-neural-supply-v4",
            catalog_seed=20260824,
            device="cpu",
        )
        world = build_world(config)
        counts = torch.bincount(world.requests.creator_id)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 8)))
        self.assertEqual(int(world.requests.request_step.min()), 0)
        self.assertEqual(int(world.requests.request_step.max()), 7)

        candidates = retrieve(world, ("popular", "geo"))
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))
        relevance_mask = response["label_masks"][:, :, 2]
        relevance_label = response["labels"][:, :, 2]
        self.assertEqual(int(relevance_mask.sum()), int(response["published"].sum()))
        self.assertTrue(torch.all(relevance_label <= relevance_mask))
        self.assertTrue(torch.all(
            response["labels"][:, :, 1] <= response["labels"][:, :, 0]
        ))

    def test_release_binds_the_promoted_linear_supply_model(self):
        root = Path(__file__).resolve().parents[2]
        release = build_posting_release(
            root,
            "reports/launches/2026-08-24-poi-posting-neural-v4-400k.json",
            "artifacts/models/poi-posting-v4",
        )
        self.assertEqual(release["schema"], "simulated-poi-posting-authority-v2")
        self.assertEqual(release["active_bundle"]["fine_model"], "linear")
        self.assertEqual(
            release["active_bundle"]["world_version"],
            "creator-neural-supply-v4",
        )
        self.assertEqual(
            release["production_readiness"],
            "hold_external_creator_and_supply_validation",
        )


if __name__ == "__main__":
    unittest.main()
