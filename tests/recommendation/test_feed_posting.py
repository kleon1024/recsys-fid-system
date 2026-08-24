from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from fid_lab.feed_posting.contracts import FeedPostingConfig
from fid_lab.feed_posting.launch import run_feed_posting_launch
from fid_lab.feed_posting.models import (
    DINRanker,
    load_bundle,
    save_bundle,
    train_models,
)
from fid_lab.feed_posting.scale import run_partitioned_feed_posting_ab
from fid_lab.feed_posting.simulation.features import candidate_features, rule_score
from fid_lab.feed_posting.simulation.response import simulate_response
from fid_lab.feed_posting.simulation.retrieval import retrieve
from fid_lab.feed_posting.simulation.world import build_world, build_world_partition


def _config(requests=600):
    return FeedPostingConfig(
        requests=requests,
        prompts=1_024,
        categories=16,
        semantic_dim=16,
        sequence_length=12,
        route_candidates=8,
        merged_candidates=12,
        exposed_candidates=4,
        train_epochs=1,
        train_batch_requests=128,
        device="cpu",
    )


class FeedPostingTest(unittest.TestCase):
    def test_partitioned_creator_ab_uses_fixed_feed_model_artifact(self):
        config = FeedPostingConfig(
            **{
                **_config(800).__dict__,
                "creators": 100,
                "catalog_seed": 20260824,
                "world_version": "creator-neural-feed-supply-v4",
            }
        )
        world = build_world(config)
        candidates = retrieve(world, ("trending", "i2i"))
        features = candidate_features(world, candidates)
        semantic = world.catalog.semantic[candidates.prompt_ids]
        response = simulate_response(world, candidates, rule_score(features))
        bundle = train_models(
            config, features, semantic, world.requests.feed_sequence,
            response["top_indices"], response["labels"],
            response["label_masks"], model_names=("linear",),
        )["linear"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.pt"
            save_bundle(bundle, path, config)
            report = run_partitioned_feed_posting_ab(
                config, path, partition_requests=317
            )
        self.assertEqual(report["requests"], 800)
        self.assertEqual(report["creators"], 100)
        self.assertEqual(
            report["creator_randomized_ab"]["publish_rate"]["estimator"],
            "cluster_randomized_ab_from_means",
        )

    def test_creator_v4_is_invariant_to_request_partition_boundaries(self):
        config = FeedPostingConfig(
            **{
                **_config(800).__dict__,
                "creators": 100,
                "catalog_seed": 20260824,
                "world_version": "creator-neural-feed-supply-v4",
            }
        )
        full = build_world(config)
        parts = (
            build_world_partition(config, 0, 317),
            build_world_partition(config, 317, 483),
        )
        self.assertTrue(torch.equal(
            full.requests.feed_sequence,
            torch.cat([part.requests.feed_sequence for part in parts]),
        ))
        full_candidates = retrieve(full, ("trending", "i2i"))
        part_candidates = [retrieve(part, ("trending", "i2i")) for part in parts]
        self.assertTrue(torch.equal(
            full_candidates.prompt_ids,
            torch.cat([part.prompt_ids for part in part_candidates]),
        ))
        full_features = candidate_features(full, full_candidates)
        part_features = [
            candidate_features(world, candidates)
            for world, candidates in zip(parts, part_candidates, strict=True)
        ]
        self.assertTrue(torch.equal(full_features, torch.cat(part_features)))

    def test_creator_neural_v4_uses_panels_and_mature_post_publish_labels(self):
        config = FeedPostingConfig(
            **{
                **_config(800).__dict__,
                "creators": 100,
                "catalog_seed": 20260824,
                "world_version": "creator-neural-feed-supply-v4",
            }
        )
        world = build_world(config)
        counts = torch.bincount(world.requests.creator_id)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 8)))
        candidates = retrieve(world, ("trending", "i2i"))
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))
        quality_mask = response["label_masks"][:, :, 3]
        quality_label = response["labels"][:, :, 3]
        risk_mask = response["label_masks"][:, :, 4]
        risk_label = response["labels"][:, :, 4]
        self.assertEqual(int(quality_mask.sum()), int(response["published"].sum()))
        self.assertTrue(torch.all(quality_label <= quality_mask))
        self.assertEqual(int(risk_mask.sum()), int(response["published"].sum()))
        self.assertTrue(torch.all(risk_label <= risk_mask))
        self.assertTrue(torch.all(response["negative"] <= response["published"]))

    def test_creator_v4_candidate_launch_reports_creator_randomized_ab(self):
        config = FeedPostingConfig(
            **{
                **_config(800).__dict__,
                "creators": 100,
                "catalog_seed": 20260824,
                "world_version": "creator-neural-feed-supply-v4",
            }
        )
        report = run_feed_posting_launch(config, candidate_only=True)
        for row in report["launches"]:
            self.assertEqual(
                row["creator_online_ab"]["publish_rate"]["estimator"],
                "creator_cluster_randomized_ab",
            )

    def test_candidate_phase_stops_before_fine_rank_training(self):
        report = run_feed_posting_launch(_config(400), candidate_only=True)
        self.assertEqual(report["schema"], "feed-posting-candidate-phase-v1")
        self.assertEqual(
            {row["stage"] for row in report["launches"]}, {"candidate"}
        )
        self.assertNotIn("models", report)

    def test_request_closure_and_behavior_cascade(self):
        config = _config(400)
        world = build_world(config)
        candidates = retrieve(world, ("trending", "i2i"))
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))

        self.assertEqual(
            candidates.prompt_ids.shape,
            (config.requests, config.merged_candidates),
        )
        ordered = candidates.prompt_ids.sort(1).values
        self.assertTrue((ordered[:, 1:] != ordered[:, :-1]).all())
        self.assertFalse(candidates.audit_oracle_recalled.all())
        self.assertTrue(torch.all(response["published"] <= response["created"]))
        self.assertTrue(torch.all(response["created"] <= response["clicked"]))
        behavioral = response["labels"][:, :, :3].sum(2) > 0
        exposed = torch.zeros_like(behavioral)
        exposed.scatter_(1, response["top_indices"], True)
        self.assertFalse((behavioral & ~exposed).any())

    def test_din_is_candidate_conditioned_and_sequence_aware(self):
        torch.manual_seed(7)
        model = DINRanker(width=14, semantic_dim=16).eval()
        features = torch.randn(3, 5, 14)
        candidates = torch.randn(3, 5, 16)
        history = torch.randn(3, 12, 16)
        with torch.inference_mode():
            original = model(features, candidates, history)["publish"]
            changed = model(features, candidates, history.flip(1) * 1.7)["publish"]
        self.assertFalse(torch.allclose(original, changed))
        self.assertFalse(torch.allclose(original[:, 0], original[:, 1]))

    def test_model_artifact_replays_exactly(self):
        config = _config()
        world = build_world(config)
        candidates = retrieve(world, ("trending", "i2i"))
        features = candidate_features(world, candidates)
        semantic = world.catalog.semantic[candidates.prompt_ids]
        response = simulate_response(world, candidates, rule_score(features))
        bundles = train_models(
            config, features, semantic, world.requests.feed_sequence,
            response["top_indices"], response["labels"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.pt"
            save_bundle(bundles["linear"], path, config)
            loaded = load_bundle(path)
            before = bundles["linear"].score(
                features[:64], semantic[:64], world.requests.feed_sequence[:64]
            )
            after = loaded.score(
                features[:64], semantic[:64], world.requests.feed_sequence[:64]
            )
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
