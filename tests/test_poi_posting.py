from __future__ import annotations

import unittest

import numpy as np

from fid_lab.poi_posting import PoiPostingConfig, PoiPostingRanker, build_dataset
from fid_lab.poi_posting.training import sampled_training_indices, tensor_batch, time_split


class PoiPostingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PoiPostingConfig(sessions=80, epochs=1)
        cls.data = build_dataset(cls.config)

    def test_contract_preserves_one_slate_per_session(self) -> None:
        counts = np.bincount(self.data.session_id)
        self.assertTrue(np.all(counts == self.config.candidates_per_session))
        self.assertGreater(int(self.data.hard_negative.sum()), 0)
        self.assertTrue(np.all(self.data.label_masks == 1.0))

    def test_easy_negative_sampling_keeps_all_important_examples(self) -> None:
        train, _, _ = time_split(self.data)
        selected, weights = sampled_training_indices(self.data, train, self.config)
        required = train[
            (self.data.labels[train, 0] > 0) | (self.data.hard_negative[train] > 0)
        ]
        self.assertTrue(set(required).issubset(set(selected)))
        self.assertEqual(len(selected), len(weights))

    def test_model_outputs_all_tasks_and_frame_attention(self) -> None:
        model = PoiPostingRanker(self.config)
        outputs = model(tensor_batch(self.data, np.arange(12)))
        for task in ("select", "publish", "relevance"):
            self.assertEqual(tuple(outputs[task].shape), (12,))
            gate = outputs[f"gate:{task}"].detach().numpy()
            np.testing.assert_allclose(gate.sum(axis=1), 1.0, atol=1e-6)
        attention = self.data.frame_attention[:12]
        np.testing.assert_allclose(attention.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
