"""Artifact replay check for request-level Retrieval V4."""

import unittest
from pathlib import Path

import torch

from fid_lab.poi_distribution.retrieval.models.bundle import load_bundle


class RetrievalV4Test(unittest.TestCase):
    def test_published_two_tower_artifact_replays_exact_pool_scores(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "artifacts/models/shared-retrieval-v4-aligned/two_tower.pt"
        features = torch.rand(128, 23)
        first = load_bundle(path).index(features)
        second = load_bundle(path).index(features)
        query = torch.rand(16, 32)
        pool = torch.randint(128, (16, 24))
        self.assertTrue(torch.equal(
            first.score_pool(query, pool), second.score_pool(query, pool)
        ))


if __name__ == "__main__":
    unittest.main()
