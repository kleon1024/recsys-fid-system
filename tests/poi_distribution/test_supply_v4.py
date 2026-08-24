from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from fid_lab.poi_posting.world import PostingWorldConfig
from fid_lab.poi_posting.world.generator import (
    build_world,
    build_world_partition,
    candidate_features,
    retrieve,
    rule_score,
    simulate_response,
)
from fid_lab.poi_posting.world.release import build_posting_release
from fid_lab.poi_posting.world.scale import (
    run_partitioned_supply_ab,
    run_partitioned_supply_replay,
)


class SupplyV4Test(unittest.TestCase):
    def test_partitioned_ab_uses_creator_means_and_fixed_artifact(self):
        config = PostingWorldConfig(
            requests=800, creators=100, cities=8, categories=4,
            items_per_cell=32, semantic_dim=8, train_epochs=1,
            world_version="creator-neural-supply-v4",
            catalog_seed=20260824, device="cpu",
        )
        root = Path(__file__).resolve().parents[2]
        report = run_partitioned_supply_ab(
            config,
            root / "artifacts/models/poi-posting-v4/seed-20260824-linear.pt",
            partition_requests=317,
        )
        self.assertEqual(report["requests"], 800)
        self.assertEqual(report["creators"], 100)
        self.assertEqual(
            report["creator_randomized_ab"]["publish_rate"]["estimator"],
            "cluster_randomized_ab_from_means",
        )

    def test_partitioned_replay_preserves_metrics_and_bounds_memory_shape(self):
        config = PostingWorldConfig(
            requests=800, creators=100, cities=8, categories=4,
            items_per_cell=32, semantic_dim=8, train_epochs=1,
            world_version="creator-neural-supply-v4",
            catalog_seed=20260824, device="cpu",
        )
        one = run_partitioned_supply_replay(config, partition_requests=800)
        many = run_partitioned_supply_replay(config, partition_requests=317)
        self.assertEqual(len(one["partitions"]), 1)
        self.assertEqual(len(many["partitions"]), 3)
        for key, value in one["metrics"].items():
            self.assertAlmostEqual(value, many["metrics"][key], places=10)

    def test_partition_assets_resume_and_recompute_invalid_signature(self):
        config = PostingWorldConfig(
            requests=800, creators=100, cities=8, categories=4,
            items_per_cell=32, semantic_dim=8, train_epochs=1,
            world_version="creator-neural-supply-v4",
            catalog_seed=20260824, device="cpu",
        )
        with TemporaryDirectory() as directory:
            partition_dir = Path(directory)
            first = run_partitioned_supply_replay(
                config, partition_requests=317, partition_dir=partition_dir
            )
            second = run_partitioned_supply_replay(
                config, partition_requests=317, partition_dir=partition_dir
            )
            corrupted = partition_dir / "part-000000000317.json"
            payload = json.loads(corrupted.read_text())
            payload["signature"] = "invalid"
            corrupted.write_text(json.dumps(payload))
            third = run_partitioned_supply_replay(
                config, partition_requests=317, partition_dir=partition_dir
            )
        self.assertEqual(first["resume"]["materialized_partitions"], 3)
        self.assertEqual(second["resume"]["reused_partitions"], 3)
        self.assertEqual(third["resume"]["materialized_partitions"], 1)
        self.assertEqual(first["metrics"], second["metrics"])

    def test_request_world_is_exactly_invariant_to_partition_boundaries(self):
        config = PostingWorldConfig(
            requests=800, creators=100, cities=8, categories=4,
            items_per_cell=32, semantic_dim=8, train_epochs=1,
            world_version="creator-neural-supply-v4",
            catalog_seed=20260824, device="cpu",
        )
        full = build_world(config)
        parts = (
            build_world_partition(config, 0, 317),
            build_world_partition(config, 317, 483),
        )
        self.assertTrue(torch.equal(
            full.requests.observed_draft,
            torch.cat([part.requests.observed_draft for part in parts]),
        ))
        full_candidates = retrieve(full, ("popular", "geo"))
        part_candidates = [retrieve(part, ("popular", "geo")) for part in parts]
        self.assertTrue(torch.equal(
            full_candidates.item_ids,
            torch.cat([part.item_ids for part in part_candidates]),
        ))
        full_features = candidate_features(full, full_candidates)
        part_features = [
            candidate_features(world, candidates)
            for world, candidates in zip(parts, part_candidates, strict=True)
        ]
        self.assertTrue(torch.equal(full_features, torch.cat(part_features)))
        full_response = simulate_response(
            full, full_candidates, rule_score(full_features)
        )
        part_response = [
            simulate_response(world, candidates, rule_score(features))
            for world, candidates, features in zip(
                parts, part_candidates, part_features, strict=True
            )
        ]
        for key in ("labels", "label_masks", "feed_stay_seconds", "negative"):
            self.assertTrue(torch.equal(
                full_response[key],
                torch.cat([response[key] for response in part_response]),
            ))

    def test_creator_panel_and_mature_labels_match_the_logging_contract(self):
        config = PostingWorldConfig(
            requests=800,
            creators=100,
            cities=8,
            categories=4,
            items_per_cell=32,
            semantic_dim=8,
            train_epochs=1,
            world_version="creator-neural-supply-v4",
            catalog_seed=20260824,
            device="cpu",
        )
        world = build_world(config)
        counts = torch.bincount(world.requests.creator_id)
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 8)))
        self.assertEqual(int(world.requests.request_step.min()), 0)
        self.assertEqual(int(world.requests.request_step.max()), 7)

        candidates = retrieve(world, ("popular", "geo"))
        features = candidate_features(world, candidates)
        response = simulate_response(world, candidates, rule_score(features))
        relevance_mask = response["label_masks"][:, :, 2]
        relevance_label = response["labels"][:, :, 2]
        self.assertEqual(int(relevance_mask.sum()), int(response["published"].sum()))
        self.assertTrue(torch.all(relevance_label <= relevance_mask))
        self.assertTrue(torch.all(
            response["labels"][:, :, 1] <= response["labels"][:, :, 0]
        ))

    def test_release_binds_the_promoted_linear_supply_model(self):
        root = Path(__file__).resolve().parents[2]
        release = build_posting_release(
            root,
            "reports/launches/2026-08-24-poi-posting-scaled-v4.json",
            "artifacts/models/poi-posting-v4",
        )
        self.assertEqual(release["schema"], "simulated-poi-posting-authority-v2")
        self.assertEqual(release["active_bundle"]["fine_model"], "linear")
        self.assertEqual(
            release["active_bundle"]["world_version"],
            "creator-neural-supply-v4",
        )
        self.assertEqual(
            release["production_readiness"],
            "hold_external_creator_and_supply_validation",
        )


if __name__ == "__main__":
    unittest.main()
