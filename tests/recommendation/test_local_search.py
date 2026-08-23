from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from fid_lab.local_search.contracts import LocalSearchConfig
from fid_lab.local_search.launch import run_local_search_seed
from fid_lab.local_search.models.ranking import load_ranker, save_ranker, train_rankers
from fid_lab.local_search.models.retrieval import (
    load_retriever,
    save_retriever,
    train_retriever,
)
from fid_lab.local_search.simulation.features import candidate_features, rule_score
from fid_lab.local_search.simulation.response import simulate_response
from fid_lab.local_search.simulation.retrieval import retrieve
from fid_lab.local_search.simulation.world import build_world


def _config(requests=800):
    return LocalSearchConfig(
        requests=requests, pois=1_024, users=requests // 2,
        categories=16, cities=16, semantic_dim=16, history_length=12,
        route_candidates=8, merged_candidates=16, exposed_candidates=5,
        train_epochs=1, train_batch_requests=128, device="cpu",
    )


class LocalSearchTest(unittest.TestCase):
    def _logging_world(self, requests=800):
        config = _config(requests)
        world = build_world(config)
        candidates = retrieve(world, ("lexical", "geo"))
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))
        return config, world, candidates, features, response

    def test_request_candidates_labels_and_observability_close(self):
        config, _, candidates, _, response = self._logging_world()
        examples = response["examples"]
        self.assertEqual(
            candidates.poi_ids.shape, (config.requests, config.merged_candidates)
        )
        ordered = candidates.poi_ids.sort(1).values
        self.assertTrue((ordered[:, 1:] != ordered[:, :-1]).all())
        self.assertFalse(candidates.audit_oracle_recalled.all())
        self.assertTrue(torch.all(response["ordered"] <= response["detail"]))
        self.assertTrue(torch.all(response["detail"] <= response["clicked"]))
        behavioral = (examples.labels[:, :, :3] > 0).any(2)
        exposed = torch.zeros_like(behavioral)
        exposed.scatter_(1, examples.exposed_indices, True)
        self.assertFalse((behavioral & ~exposed).any())
        self.assertTrue((examples.position_propensity > 0).all())
        self.assertTrue((examples.label_masks[:, :, 3] == 0).any())

    def test_two_tower_uses_exposed_negatives_and_replays(self):
        _, world, _, _, response = self._logging_world()
        bundle = train_retriever(world.config, world, response)
        self.assertGreater(bundle.offline["training_pairs"], 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.pt"
            save_retriever(bundle, path)
            loaded = load_retriever(path)
            features = torch.zeros(16, bundle.model.query_width)
            with torch.inference_mode():
                before = bundle.model.encode_query(features)
                after = loaded.model.encode_query(features)
        self.assertTrue(torch.equal(before, after))

    def test_ranker_artifacts_replay_and_candidate_phase_stops(self):
        config, world, candidates, features, response = self._logging_world(1_200)
        rankers = train_rankers(config, world, candidates, features, response)
        semantic = world.catalog.semantic[candidates.poi_ids]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("linear", "xgboost_pairwise"):
                suffix = ".json" if name == "xgboost_pairwise" else ".pt"
                path = root / f"{name}{suffix}"
                save_ranker(rankers[name], path, config)
                loaded = load_ranker(path)
                before = rankers[name].score(
                    features[:32], semantic[:32],
                    world.requests.history_sequence[:32],
                )
                after = loaded.score(
                    features[:32], semantic[:32],
                    world.requests.history_sequence[:32],
                )
                self.assertTrue(torch.allclose(before, after, atol=1e-7))
        phase = run_local_search_seed(config, candidate_only=True)
        self.assertEqual(phase["schema"], "local-search-retrieval-phase-v1")
        self.assertEqual({row["stage"] for row in phase["launches"]}, {"retrieval"})


if __name__ == "__main__":
    unittest.main()
