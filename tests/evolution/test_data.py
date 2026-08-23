from __future__ import annotations

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
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
from fid_lab.evolution.data.request_dataset import (
    build_request_candidate_dataset,
    dataset_tables,
    materialize_dataset,
)
from fid_lab.evolution.data.sampling import (
    expected_sampling_counts,
    mixed_negative_sample,
    negative_source_counts,
)
from fid_lab.scale import FeedTensorDataset, ScaleConfig, build_scale_dataset
from fid_lab.scale.dataset import tensorflow_generator
from fid_lab.scale.diagnostics import diagnose_auc_without_lift
from fid_lab.simulation.contracts import SimulationConfig
from fid_lab.simulation.environment import build_catalog
from fid_lab.simulation.experiment import build_feed_joiner
from fid_lab.simulation.policies import HeuristicPolicy
from fid_lab.simulation.population import run_population


def decision(request: str, video: int, event_time: int = 100, *, viewer: int = 1,
             exposed: bool = True, category: int = 3, city: int = 2) -> StageDecision:
    return StageDecision(
        request,
        viewer,
        video + 100,
        video,
        9,
        event_time,
        category,
        city,
        tuple(range(6)),
        (0.0,) * 10,
        tuple((0.0,) * 8 for _ in range(24)),
        "ann",
        0.2,
        0.7,
        1,
        exposed,
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

    def test_nonlinear_signal_world_is_versioned_and_deterministic(self) -> None:
        config = ScaleConfig(
            main_impressions=20_000,
            anchor_rate=0.02,
            seed=9,
            signal_version="heterogeneous-nonlinear-v2",
        )
        first = build_scale_dataset(config)
        second = build_scale_dataset(config)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_allclose(first.label_probabilities, second.label_probabilities)
        baseline = build_scale_dataset(
            ScaleConfig(main_impressions=20_000, anchor_rate=0.02, seed=9)
        )
        self.assertFalse(
            np.allclose(first.label_probabilities, baseline.label_probabilities)
        )
        with self.assertRaisesRegex(ValueError, "unsupported signal version"):
            ScaleConfig(signal_version="unknown")


class SamplingAndJoinerTest(unittest.TestCase):
    def test_sampling_correction_uses_source_expected_counts(self) -> None:
        counts = negative_source_counts(20)
        probabilities = np.concatenate(
            (
                np.full(counts["in_batch"], 0.1),
                np.full(counts["hard"], 0.2),
                np.full(counts["random"], 0.05),
            )
        )[None]
        corrected = expected_sampling_counts(probabilities, 20)
        np.testing.assert_allclose(
            corrected[0, : counts["in_batch"]], counts["in_batch"] * 0.1
        )
        np.testing.assert_allclose(corrected[0, -counts["random"] :], 0.15)

    def test_request_candidate_dataset_closes_every_stage_and_label(self) -> None:
        config = SimulationConfig(
            users=4,
            items=300,
            candidates=10,
            joiner_users=4,
            seed=19,
            signal_version="heterogeneous-nonlinear-v2",
        )
        catalog = build_catalog(config)
        policy = HeuristicPolicy()
        trajectories = run_population(config, catalog, policy, range(config.users))
        assigned = np.zeros(config.users, dtype=bool)
        joined = build_feed_joiner(
            config, catalog, trajectories, (policy, policy), assigned
        )
        dataset = build_request_candidate_dataset(
            trajectories,
            catalog,
            joined,
            {"model": policy.name, "feature": "stateful-v2"},
        )
        self.assertEqual(
            len(dataset.candidates),
            sum(row.recall_count for value in trajectories for row in value.rows),
        )
        by_request = {
            request.request_id: [
                candidate
                for candidate in dataset.candidates
                if candidate.request_id == request.request_id
            ]
            for request in dataset.requests
        }
        for request_id, candidates in by_request.items():
            self.assertEqual(sum(value.coarse_pass for value in candidates), 10)
            self.assertEqual(sum(value.exposed_position == 1 for value in candidates), 1)
            self.assertEqual(len({value.candidate_id for value in candidates}), len(candidates))
            self.assertTrue(all(value.recall_routes for value in candidates))
        self.assertEqual(
            dataset.stage_attribution["requests"],
            sum(
                dataset.stage_attribution[name]
                for name in (
                    "recall_miss",
                    "coarse_miss",
                    "fine_rank_miss",
                    "mix_rank_miss",
                    "served_oracle",
                )
            ),
        )
        exposed_labels = [
            label for label in dataset.labels if label.exchanged_lt_components
        ]
        self.assertEqual(len(exposed_labels), len(dataset.requests))
        self.assertFalse(dataset.requests[0].sequence_snapshot)
        self.assertEqual(
            set(dataset_tables(dataset)),
            {"requests", "candidate_decisions", "mature_labels"},
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = materialize_dataset(dataset, output)
            self.assertEqual(manifest["schema"], "request-candidate-v1")
            self.assertEqual(
                manifest["tables"]["candidate_decisions"]["rows"],
                len(dataset.candidates),
            )
            self.assertTrue((output / "manifest.json").exists())

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
        probability = {
            source: {value.sampling_probability for value in samples if value.source == source}
            for source in ("in_batch", "hard", "random")
        }
        self.assertEqual(probability, {
            "in_batch": {0.01}, "hard": {0.02}, "random": {0.01},
        })

    def test_recall_negatives_preserve_source_semantics(self) -> None:
        decisions = [
            decision("r1", 10, viewer=1),
            decision("r1", 11, viewer=1, exposed=False),
            decision("r1", 12, viewer=1, exposed=False, category=4, city=5),
            decision("r2", 20, viewer=2),
            decision("r2", 21, viewer=2, exposed=False),
        ]
        actions = [
            ActionEvent("a1", "r1", 10, 9, "long_view", 120, 121),
            ActionEvent("a2", "r2", 20, 9, "long_view", 120, 121),
        ]
        report = EvolutionJoiner().build(
            decisions, actions, [], [], [], watermark=700_000
        )
        self.assertEqual(len(report.recall), 2)
        first = report.recall[0]
        by_source = {
            source: {value.item_id for value in first.negatives if value.source == source}
            for source in ("in_batch", "hard", "random")
        }
        self.assertEqual(by_source["in_batch"], {20})
        self.assertEqual(by_source["hard"], {11})
        self.assertNotIn(10, by_source["random"])

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
