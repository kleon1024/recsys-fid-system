from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np
import torch

from fid_lab.data import encode_rows, make_synthetic_rows
from fid_lab.fid import FidCodec, FidVersion
from fid_lab.models import DeepFM, ThreeTower, WideDeep
from fid_lab.schema import DEFAULT_SCHEMA, FeatureRegistry, FeatureSpec


class FidCodecTest(unittest.TestCase):
    def test_v1_round_trip_and_mask(self) -> None:
        codec = FidCodec(FidVersion.V1)
        fid = codec.pack(1023, 1 << 54 | 17)
        self.assertEqual(codec.unpack(fid), (1023, 17))

    def test_v2_round_trip_reserved_bit_and_bounds(self) -> None:
        codec = FidCodec(FidVersion.V2)
        fid = codec.pack(32767, 1 << 48 | 23)
        self.assertEqual(codec.unpack(fid), (32767, 23))
        with self.assertRaises(ValueError):
            codec.pack(32768, 1)
        with self.assertRaises(ValueError):
            codec.unpack(1 << 63)

    def test_v1_to_v2_truncates_six_signature_bits(self) -> None:
        v1 = FidCodec(FidVersion.V1).pack(8, (42 << 48) | 19)
        v2 = FidCodec.convert_v1_to_v2(v1)
        self.assertEqual(FidCodec(FidVersion.V2).unpack(v2), (8, 19))

    def test_signature_is_stable_across_processes(self) -> None:
        expected = FidCodec().signature("user_id", "123")
        command = [
            sys.executable,
            "-c",
            "from fid_lab.fid import FidCodec; print(FidCodec().signature('user_id', '123'))",
        ]
        actual = int(subprocess.check_output(command, text=True).strip())
        self.assertEqual(actual, expected)


class RegistryTest(unittest.TestCase):
    def test_cross_encoding_is_unambiguous(self) -> None:
        registry = FeatureRegistry([FeatureSpec("cross", 1, "cross", ("a", "b"), 16)])
        codec = FidCodec()
        first = registry.encode_row({"a": "ab", "b": "c"}, codec)
        second = registry.encode_row({"a": "a", "b": "bc"}, codec)
        self.assertNotEqual(first, second)

    def test_dataset_has_one_encoding_for_all_models(self) -> None:
        rows, labels = make_synthetic_rows(32, seed=4)
        encoded = encode_rows(rows, labels, DEFAULT_SCHEMA, FidCodec())
        self.assertEqual(encoded.bucket_ids.shape, (32, len(DEFAULT_SCHEMA.specs)))
        self.assertEqual(encoded.fids.dtype, np.uint64)
        self.assertTrue(np.array_equal(encoded.labels, labels))


class ModelContractTest(unittest.TestCase):
    def test_every_neural_model_returns_one_logit_per_example(self) -> None:
        bucket_sizes = [spec.buckets for spec in DEFAULT_SCHEMA.specs]
        features = torch.zeros((4, len(bucket_sizes)), dtype=torch.long)
        models = [
            WideDeep(bucket_sizes),
            DeepFM(bucket_sizes),
            ThreeTower(bucket_sizes, DEFAULT_SCHEMA),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model(features).shape, (4,))


if __name__ == "__main__":
    unittest.main()
