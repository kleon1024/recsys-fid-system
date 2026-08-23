from __future__ import annotations

import unittest

import numpy as np
import torch

from fid_lab.surfaces import SURFACE_SPECS, build_surface_model


class SurfaceModelTest(unittest.TestCase):
    def test_every_surface_has_distinct_contract_and_valid_gate(self) -> None:
        self.assertEqual(
            set(SURFACE_SPECS),
            {"feed_poi_video", "poi_map_detail", "ymal", "product", "review"},
        )
        for spec in SURFACE_SPECS.values():
            model = build_surface_model(spec)
            sequence = torch.randn(5, 24, 8) if spec.name == "feed_poi_video" else None
            outputs = model(torch.randn(5, len(spec.features)), sequence)
            for task in spec.task_names:
                self.assertEqual(tuple(outputs[task].shape), (5,))
                gate_key = f"gate:{task}"
                if gate_key in outputs:
                    gate = outputs[gate_key].detach().numpy()
                    np.testing.assert_allclose(gate.sum(axis=1), 1.0, atol=1e-6)

    def test_surface_labels_are_not_reused(self) -> None:
        task_sets = {
            name: set(spec.task_names) for name, spec in SURFACE_SPECS.items()
        }
        self.assertNotEqual(task_sets["feed_poi_video"], task_sets["product"])
        self.assertNotEqual(task_sets["poi_map_detail"], task_sets["review"])


if __name__ == "__main__":
    unittest.main()
