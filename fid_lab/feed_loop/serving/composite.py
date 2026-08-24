"""One serving policy composing main Feed and POI/Local model artifacts."""

from __future__ import annotations

import torch

from .contracts import CandidateScoreBundle, CompositeValueTreeConfig
from .value_tree import CompositeValueTree, request_standardize
from ..scale.artifact.features import build_tensor_features
from ..tensor_cascade import (
    _fine_score, coarse_rank, materialize_selected,
)
from ..tensor_policies import PERSONALIZED


class CompositeTensorPolicy:
    eligible_fraction = 1.0
    observation_noise = PERSONALIZED.observation_noise
    local_observation_noise = PERSONALIZED.local_observation_noise
    realtime_interest_rate = PERSONALIZED.realtime_interest_rate
    multi_queue = False

    def __init__(self, feed_policy, local_bundle, config=None):
        self.feed_policy = feed_policy
        self.local_bundle = local_bundle
        self.config = config or CompositeValueTreeConfig()
        if feed_policy.blend_weight != self.config.feed_residual_weight:
            raise ValueError("composite tree and Feed release blend differ")
        if feed_policy.base_tolerance != self.config.base_tolerance:
            raise ValueError("composite tree and Feed release guard differ")
        self.value_tree = CompositeValueTree(feed_policy, self.config)
        self.name = (
            f"unified_{feed_policy.name}_{local_bundle.name}_local_"
            f"{self.config.local_fine_weight:g}"
        )

    def describe(self):
        return {
            "name": self.name,
            "feed_model": self.feed_policy.describe(),
            "local_model": self.local_bundle.name,
            "value_tree": self.config.manifest(),
        }

    def _bundle(self, features, sequence, base, candidates):
        shape = features.shape[:2]
        feed = self.feed_policy.predict_tasks(features, sequence)
        local = {
            name: value.reshape(shape)
            for name, value in self.local_bundle.probabilities(
                features.flatten(0, 1)
            ).items()
        }
        return CandidateScoreBundle(
            base, feed, local,
            {"ad": candidates["ad_value"], "live": candidates["live_value"]},
            {
                "feed": self.feed_policy.model_name,
                "local": self.local_bundle.name,
                "value_tree": self.config.version,
            },
        )

    @torch.inference_mode()
    def select_candidate(self, user_ids, state, candidates, device, step, config):
        if "ranking_behavior_sequence" not in state:
            raise ValueError("composite serving requires the online sequence")
        features = build_tensor_features(config, user_ids, state, candidates, step)
        base, affinity = _fine_score(
            PERSONALIZED, state["eligible"], user_ids, state, candidates
        )
        bundle = self._bundle(
            features, state["ranking_behavior_sequence"], base, candidates
        )
        score, components = self.value_tree.evaluate(bundle, features, candidates)
        coarse_base, _, _ = coarse_rank(PERSONALIZED, affinity, candidates)
        keep = min(self.config.local_coarse_keep, coarse_base.shape[1])
        coarse = coarse_base + self.config.local_coarse_weight * request_standardize(
            components["local_model_value"]
        )
        indices = torch.topk(coarse, keep, dim=1).indices
        coarse_mask = torch.zeros_like(coarse, dtype=torch.bool)
        coarse_mask.scatter_(1, indices, True)
        eligible = coarse_mask & (
            base >= base.max(dim=1, keepdim=True).values
            - self.config.base_tolerance
        )
        served = score.masked_fill(~eligible, -1e9)
        choice = served.argmax(dim=1)
        selected = materialize_selected(
            self, user_ids, state, candidates, choice, choice, served, served,
            coarse, coarse_mask, keep, device,
        )
        batch = torch.arange(len(choice), device=device)
        selected.update({
            name: value[batch, choice] for name, value in components.items()
        })
        selected["score_bundle_versions"] = bundle.model_versions
        return selected
