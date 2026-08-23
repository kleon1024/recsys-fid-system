import unittest

import numpy as np
import torch

from fid_lab.feed_loop.scale.small_effect_ab import run_small_effect_ab
from fid_lab.simulation.experimentation import (
    Experiment,
    ExperimentLayer,
    FeedParameters,
    Variant,
    assign_binary_torch,
    assign_layer_numpy,
    assign_layers,
)


class ExperimentationTest(unittest.TestCase):
    @staticmethod
    def _layers():
        return (
            ExperimentLayer(
                "rank",
                "rank-layer",
                (
                    Experiment(
                        "fine-model",
                        (
                            Variant("control", 0.25, {"fine_model": "lr_v1"}),
                            Variant("treatment", 0.25, {"fine_model": "deepfm_v1"}),
                        ),
                    ),
                ),
            ),
            ExperimentLayer(
                "value",
                "value-layer",
                (
                    Experiment(
                        "diversity",
                        (
                            Variant("control", 0.25, {"diversity_strength": 0.0}),
                            Variant("treatment", 0.25, {"diversity_strength": 0.1}),
                        ),
                    ),
                ),
            ),
        )

    def test_overlapping_layers_are_balanced_and_full_chain_parameterized(self):
        layers = self._layers()
        assignments = [
            assign_layers(user, layers, FeedParameters()) for user in range(20_000)
        ]
        rank_treatment = sum(
            value["assignments"]["rank"]["variant"] == "treatment"
            for value in assignments
        )
        value_treatment = sum(
            value["assignments"]["value"]["variant"] == "treatment"
            for value in assignments
        )
        both = sum(
            value["assignments"]["rank"]["variant"] == "treatment"
            and value["assignments"]["value"]["variant"] == "treatment"
            for value in assignments
        )
        self.assertTrue(0.23 < rank_treatment / len(assignments) < 0.27)
        self.assertTrue(0.23 < value_treatment / len(assignments) < 0.27)
        self.assertTrue(0.05 < both / len(assignments) < 0.075)
        self.assertEqual(assignments[0], assign_layers(0, layers, FeedParameters()))
        self.assertIn("model_manifest", assignments[0]["parameters"])
        ids = np.arange(1_000_000, dtype="uint64")
        rank_cells, _ = assign_layer_numpy(ids, layers[0])
        value_cells, _ = assign_layer_numpy(ids, layers[1])
        self.assertTrue(0.24 < (rank_cells == 1).mean() < 0.26)
        self.assertLess(abs(np.corrcoef(rank_cells == 1, value_cells == 1)[0, 1]), 0.01)

    def test_vectorized_ab_recovers_effect_and_cuped_reduces_variance(self):
        report = run_small_effect_ab(users=200_000, relative_effects=(0.01,))[0]
        self.assertLess(abs(report["cuped_relative_lift"] - 0.01), 0.005)
        self.assertGreater(report["variance_reduction"], 0.25)
        self.assertTrue(report["truth_inside_cuped_interval"])

    def test_gpu_binary_assignment_is_orthogonal_to_uid_cohorts(self):
        identifiers = torch.arange(200_000)
        treatment = assign_binary_torch(identifiers).numpy()
        phase = np.remainder(np.arange(len(identifiers)), 997) / 997.0
        cohort = np.sin(2.0 * np.pi * phase)
        self.assertLess(abs(treatment.mean() - 0.5), 0.005)
        self.assertLess(abs(np.corrcoef(treatment, cohort)[0, 1]), 0.01)

    def test_two_layers_cannot_own_the_same_parameter(self):
        layers = tuple(
            ExperimentLayer(
                name,
                name,
                (
                    Experiment(
                        name,
                        (Variant("treatment", 0.5, {"fine_model": name}),),
                    ),
                ),
            )
            for name in ("layer_a", "layer_b")
        )
        with self.assertRaisesRegex(ValueError, "owned by both"):
            assign_layers(1, layers, FeedParameters())


if __name__ == "__main__":
    unittest.main()
