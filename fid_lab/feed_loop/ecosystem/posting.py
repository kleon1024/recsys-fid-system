"""Adapt creator ecosystem state into the Feed Posting request authority."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from ...feed_posting.contracts import FeedPostingConfig
from ...feed_posting.models import load_bundle
from ...feed_posting.serving import blend_score, policy_name
from ...feed_posting.simulation.features import candidate_features, rule_score
from ...feed_posting.simulation.response import simulate_response
from ...feed_posting.simulation.retrieval import retrieve
from ...feed_posting.simulation.world import (
    FeedPostingWorld,
    build_world_partition,
    normalize,
)


class FeedPostingIntervention:
    """Score prompts using point-in-time creator state and a frozen artifact."""

    def __init__(
        self, config: FeedPostingConfig, model_path: Path, blend: float,
        batch_creators: int = 25_000, blend_mode: str = "legacy_convex",
    ):
        if config.world_version != "creator-neural-feed-supply-v4":
            raise ValueError("ecosystem posting intervention requires creator V4")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("ecosystem posting blend must be between zero and one")
        if batch_creators < 1:
            raise ValueError("ecosystem posting batch must be positive")
        self.config = config
        self.bundle = load_bundle(model_path, config.device)
        self.blend = blend
        self.blend_mode = blend_mode
        self.batch_creators = batch_creators
        self.name = policy_name(self.bundle.name, blend, blend_mode)

    def _adapt_world(self, world, population, feedback):
        requests = world.requests
        if population.topic.shape[1] == self.config.semantic_dim:
            creative_topic = normalize(population.topic)
        elif population.topic.shape[1] <= len(world.category_basis):
            creative_topic = normalize(
                population.topic
                @ world.category_basis[: population.topic.shape[1]]
            )
        else:
            raise ValueError("creator topic schema cannot map to posting semantics")
        exposure = feedback.exposures.clamp_min(1.0)
        stay = feedback.stay / exposure
        engagement = feedback.engagement / exposure
        observed_profile = normalize(
            0.72 * creative_topic + 0.28 * requests.observed_profile
        )
        sequence = requests.feed_sequence.clone()
        sequence[:, -1] = normalize(
            creative_topic + 0.10 * requests.feed_sequence[:, -1]
        )
        sequence_feedback = requests.sequence_feedback.clone()
        sequence_feedback[:, -1] = torch.sigmoid(
            stay / 20.0 + 0.20 * engagement - 0.30 * population.fatigue
        )
        recency = torch.linspace(
            0.25, 1.0, self.config.sequence_length,
            device=sequence.device,
        )[None, :]
        weight = sequence_feedback * recency
        sequence_summary = normalize(
            (sequence * weight[:, :, None]).sum(1)
            / weight.sum(1, keepdim=True).clamp_min(1e-6)
        )
        experience = torch.sigmoid(
            torch.log1p(population.cumulative_posts) - 1.5
        )
        adapted = replace(
            requests,
            latent_intent=normalize(
                0.65 * creative_topic + 0.35 * requests.latent_intent
            ),
            observed_profile=observed_profile,
            feed_sequence=sequence,
            sequence_feedback=sequence_feedback,
            sequence_summary=sequence_summary,
            recent_category=population.topic.argmax(1),
            creator_history=creative_topic,
            creator_category=population.topic.argmax(1),
            experience=experience,
            fatigue=population.fatigue,
            activity=population.motivation,
            outside_preference=(
                0.75 - population.motivation + 0.65 * population.fatigue
            ),
        )
        return FeedPostingWorld(
            world.config, world.catalog, adapted, world.category_basis,
            world.teacher,
        )

    @staticmethod
    def _slice_state(state, start, stop):
        return type(state)(**{
            name: value[start:stop]
            for name, value in vars(state).items()
        })

    def _respond_partition(self, population, feedback, day, creator_offset):
        start = day * self.config.creators + creator_offset
        world = build_world_partition(
            self.config, start, len(population.creator_ids)
        )
        world = self._adapt_world(world, population, feedback)
        candidates = retrieve(world, ("trending", "i2i"))
        features = candidate_features(world, candidates)
        semantic = world.catalog.semantic[candidates.prompt_ids]
        baseline = rule_score(features)
        learned = self.bundle.score(
            features, semantic, world.requests.feed_sequence
        )
        scores = blend_score(
            baseline, learned, self.blend, self.blend_mode
        )
        response = simulate_response(world, candidates, scores)
        response["published"] &= population.active
        response["created"] &= population.active
        response["clicked"] &= population.active
        response["negative"] &= population.active
        return {
            name: response[name]
            for name in (
                "clicked", "created", "published", "quality_potential",
                "content_risk", "negative",
            )
        }

    def respond(self, population, feedback, day):
        if len(population.creator_ids) != self.config.creators:
            raise ValueError("creator population and posting config differ")
        parts = []
        for start in range(0, len(population.creator_ids), self.batch_creators):
            stop = min(start + self.batch_creators, len(population.creator_ids))
            parts.append(self._respond_partition(
                self._slice_state(population, start, stop),
                self._slice_state(feedback, start, stop),
                day, start,
            ))
        return {
            name: torch.cat([part[name] for part in parts])
            for name in parts[0]
        }
