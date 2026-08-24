from __future__ import annotations

from pathlib import Path
import unittest

import torch

from fid_lab.feed_loop.ecosystem.creators import (
    CreatorFeedback,
    CreatorPopulation,
)
from fid_lab.feed_loop.ecosystem.posting import FeedPostingIntervention
from fid_lab.feed_posting.contracts import FeedPostingConfig


class FeedPostingEcosystemTest(unittest.TestCase):
    def test_posting_intervention_consumes_point_in_time_creator_state(self):
        creators, width = 100, 32
        ids = torch.arange(creators)
        topic = torch.nn.functional.normalize(torch.randn(creators, width), dim=1)
        population = CreatorPopulation(
            creator_ids=ids,
            mixture=torch.remainder(ids, 4),
            region=torch.remainder(ids, 10),
            active=torch.ones(creators, dtype=torch.bool),
            motivation=torch.full((creators,), 0.65),
            fatigue=torch.full((creators,), 0.20),
            quality=torch.full((creators,), 0.55),
            topic=topic,
            expected_exposure=torch.full((creators,), 8.0),
            cumulative_posts=torch.ones(creators),
            cumulative_retained_days=torch.ones(creators),
        )
        feedback = CreatorFeedback.empty(creators, "cpu")
        feedback.exposures.fill_(4.0)
        feedback.stay.fill_(40.0)
        feedback.engagement.fill_(0.5)
        config = FeedPostingConfig(
            requests=200, creators=creators, prompts=1_024, categories=32,
            semantic_dim=width, sequence_length=64, route_candidates=8,
            merged_candidates=12, exposed_candidates=4,
            world_version="creator-neural-feed-supply-v4", device="cpu",
        )
        root = Path(__file__).resolve().parents[2]
        intervention = FeedPostingIntervention(
            config,
            root / "artifacts/models/feed-posting-v42/seed-20260824-din.pt",
            0.20, batch_creators=37,
        )
        response = intervention.respond(population, feedback, day=0)
        self.assertEqual(response["published"].shape, (creators,))
        self.assertTrue(torch.all(response["published"] <= response["created"]))
        self.assertTrue(torch.all(response["created"] <= response["clicked"]))
        self.assertEqual(intervention.name, "din_blend_0.20")


if __name__ == "__main__":
    unittest.main()
