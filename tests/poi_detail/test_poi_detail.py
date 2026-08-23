from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from fid_lab.poi_detail.contracts import PoiDetailConfig
from fid_lab.poi_detail.models.architectures import build_family
from fid_lab.poi_detail.models.training import (
    load_bundle,
    save_bundle,
    train_families,
)
from fid_lab.poi_detail.simulation.candidates import build_candidates
from fid_lab.poi_detail.simulation.features import (
    candidate_features,
    candidate_semantic,
    rule_score,
)
from fid_lab.poi_detail.simulation.response import simulate_response
from fid_lab.poi_detail.simulation.world import build_world


def _config(requests=600):
    return PoiDetailConfig(
        requests=requests,
        users=requests // 2,
        entities_per_module=1_024,
        categories=16,
        semantic_dim=16,
        history_length=12,
        train_epochs=1,
        train_batch_requests=128,
        device="cpu",
    )


class PoiDetailTest(unittest.TestCase):
    def _logging_world(self, requests=600):
        config = _config(requests)
        world = build_world(config)
        candidates = build_candidates(world)
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))
        return config, world, candidates, features, response

    def test_module_quota_and_cascade_contract(self):
        config, _, candidates, _, response = self._logging_world()
        self.assertEqual(candidates.entity_ids.shape, (config.requests, 24))
        self.assertEqual(response["top_indices"].shape, (config.requests, 8))
        selected_kinds = candidates.module_kind.gather(
            1, response["top_indices"]
        )
        self.assertTrue(torch.all((selected_kinds == 0).sum(1) == 4))
        self.assertTrue(torch.all((selected_kinds == 1).sum(1) == 2))
        self.assertTrue(torch.all((selected_kinds == 2).sum(1) == 2))
        self.assertTrue(torch.all(response["transaction"] <= response["deep"]))
        self.assertTrue(torch.all(response["deep"] <= response["clicked"]))

        exposed = torch.zeros_like(candidates.entity_ids, dtype=torch.bool)
        exposed.scatter_(1, response["top_indices"], True)
        labeled = (response["labels"] > 0).any(2)
        self.assertFalse((labeled & ~exposed).any())
        review = candidates.module_kind == 2
        self.assertTrue((response["label_masks"][:, :, 2][review] == 0).all())

    def test_module_families_do_not_share_weights(self):
        model = build_family("linear", 12, 16)
        parameter_ids = [
            {id(value) for value in module.parameters()}
            for module in model.modules_by_kind
        ]
        self.assertTrue(parameter_ids[0].isdisjoint(parameter_ids[1]))
        self.assertTrue(parameter_ids[1].isdisjoint(parameter_ids[2]))

    def test_ranker_artifact_replays_exactly(self):
        config, world, candidates, features, response = self._logging_world(800)
        semantic = candidate_semantic(world, candidates)
        bundles = train_families(
            config, world, candidates, features, semantic, response
        )
        bundle = bundles["linear"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.pt"
            save_bundle(bundle, path, config)
            loaded = load_bundle(path)
            before = bundle.score(
                features[:32], semantic[:32],
                world.requests.history_sequence[:32],
                candidates.module_kind[:32],
            )
            after = loaded.score(
                features[:32], semantic[:32],
                world.requests.history_sequence[:32],
                candidates.module_kind[:32],
            )
        self.assertTrue(torch.equal(before, after))

    def test_fixed_quota_is_part_of_v1_contract(self):
        with self.assertRaisesRegex(ValueError, "4/2/2"):
            PoiDetailConfig(exposed_related=3, exposed_product=3)


if __name__ == "__main__":
    unittest.main()
