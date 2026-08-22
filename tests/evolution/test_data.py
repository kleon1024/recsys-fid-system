from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from fid_lab.evolution.data.contracts import (
    ActionEvent,
    CommerceEvent,
    OutboundClick,
    PixelEvent,
    StageDecision,
)
from fid_lab.evolution.data.joiner import EvolutionJoiner
from fid_lab.evolution.data.sampling import mixed_negative_sample
from fid_lab.scale import FeedTensorDataset, ScaleConfig, build_scale_dataset
from fid_lab.scale.dataset import tensorflow_generator
from fid_lab.scale.diagnostics import diagnose_auc_without_lift


def decision(request: str, video: int, event_time: int = 100) -> StageDecision:
    return StageDecision(
        request,
        1,
        video + 100,
        video,
        9,
        event_time,
        3,
        2,
        tuple(range(6)),
        (0.0,) * 10,
        tuple((0.0,) * 8 for _ in range(24)),
        "ann",
        0.2,
        0.7,
        1,
        True,
        True,
        {"recall": 0.4, "coarse": 0.5, "fine": 0.7, "value": 0.6},
        {"features": "v1", "model": "v1", "index": "v1"},
    )


class ScaleDataTest(unittest.TestCase):
    def test_long_tail_distribution_and_tensor_contract(self) -> None:
        dataset = build_scale_dataset(
            ScaleConfig(main_impressions=20_000, anchor_rate=0.02, seed=7)
        )
        self.assertGreater(dataset.examples, 300)
        self.assertLess(dataset.examples, 500)
        sample = FeedTensorDataset(dataset)[0]
        self.assertEqual(tuple(sample["sparse_fids"].shape), (6,))
        self.assertEqual(tuple(sample["behavior_sequence"].shape), (24, 8))
        self.assertEqual(tuple(sample["label_masks"].shape), (6,))
        tensorflow_sample = next(tensorflow_generator(dataset))
        self.assertEqual(set(sample), set(tensorflow_sample))

    def test_auc_without_lift_diagnostic_orders_chain_failures(self) -> None:
        report = diagnose_auc_without_lift(
            srm_p_value=0.001,
            trigger_rate=0.005,
            score_replay_delta=0.2,
            coarse_positive_pass=0.8,
            calibration_error=0.1,
            experiment_power=0.4,
        )
        self.assertEqual(report.likely_causes[0], "sample_ratio_mismatch")
        self.assertIn("cascade_opportunity_loss", report.likely_causes)


class SamplingAndJoinerTest(unittest.TestCase):
    def test_negative_mix_is_exact_and_probability_carrying(self) -> None:
        samples = mixed_negative_sample(
            tuple(range(100)), tuple(range(100, 150)), tuple(range(150, 250)), 20, 4
        )
        self.assertEqual(Counter(value.source for value in samples), {
            "in_batch": 12,
            "hard": 5,
            "random": 3,
        })
        self.assertTrue(all(0.0 < value.sampling_probability <= 1.0 for value in samples))

    def test_joiner_closes_commerce_and_fractional_pixel_labels(self) -> None:
        decisions = [decision("r1", 10), decision("r2", 11, 200)]
        actions = [ActionEvent("a", "r1", 10, 9, "long_view", 120, 121)]
        commerce = [CommerceEvent("p", "r1", 10, 9, "payment", "o", "pay", 400, 401)]
        clicks = [
            OutboundClick("c1", "r1", 10, 9, "u", 7, 1_000),
            OutboundClick("c2", "r2", 11, 9, "u", 7, 2_000),
        ]
        pixels = [PixelEvent("px", "cv", "u", 7, 3_000, 3_001)]
        report = EvolutionJoiner().build(
            decisions, actions, commerce, clicks, pixels, watermark=700_000
        )
        self.assertEqual(len(report.fine), 2)
        self.assertEqual(report.fine[0].labels["payment"], 1.0)
        weights = [value.labels["pixel_conversion"] for value in report.fine]
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(report.attribution.attributed_weight, 1.0)
        self.assertTrue(all(value.label_masks["pixel_conversion"] for value in report.fine))


if __name__ == "__main__":
    unittest.main()
