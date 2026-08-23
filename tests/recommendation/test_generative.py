from __future__ import annotations

import unittest

import numpy as np

from fid_lab.generative import GenerativeRetriever, SemanticIdIndex


class GenerativeRecommendationTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(9)
        self.item_ids = np.arange(80)
        self.embeddings = rng.normal(size=(80, 12))
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index = SemanticIdIndex.fit(
            self.item_ids, self.embeddings, levels=2, codebook_size=4, seed=9
        )

    def test_semantic_ids_are_unique_and_resolve_to_items(self) -> None:
        self.assertEqual(len(self.index.codes), len(self.item_ids))
        self.assertEqual(len(set(self.index.codes.values())), len(self.item_ids))
        for item_id, code in self.index.codes.items():
            self.assertEqual(self.index.item_for_code(code), item_id)

    def test_constrained_beam_returns_only_valid_unique_items(self) -> None:
        results = GenerativeRetriever(self.index).retrieve(self.embeddings[0], limit=10, beam_size=20)
        self.assertEqual(len(results), 10)
        self.assertEqual(len({result.item_id for result in results}), 10)
        self.assertTrue(all(result.semantic_id in self.index.codes.values() for result in results))
        self.assertEqual(results[0].item_id, 0)


if __name__ == "__main__":
    unittest.main()
