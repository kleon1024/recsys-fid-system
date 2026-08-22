from __future__ import annotations

import unittest

import numpy as np

from fid_lab.media import MediaAsset, MediaFeatureMaterializer


class MediaMaterializerTest(unittest.TestCase):
    def test_materialization_is_versioned_normalized_and_deterministic(self) -> None:
        frames = np.eye(4, 8, dtype=np.float32)
        text = np.ones(8, dtype=np.float32) / 8**0.5
        asset = MediaAsset(7, frames, text, 100)
        materializer = MediaFeatureMaterializer(8, output_dim=12)
        first = materializer.materialize(asset)
        second = materializer.materialize(asset)
        np.testing.assert_array_equal(first.content_embedding, second.content_embedding)
        self.assertAlmostEqual(float(np.linalg.norm(first.content_embedding)), 1.0)
        self.assertEqual(first.encoder_version, "media-encoder-v1")
        self.assertEqual(first.content_hash, second.content_hash)


if __name__ == "__main__":
    unittest.main()
