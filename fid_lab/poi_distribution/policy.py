"""GPU serving adapter that isolates coarse, fine, and mix model changes."""

from __future__ import annotations

import torch

from ..feed_loop.scale.artifact.features import build_tensor_features
from ..feed_loop.tensor_cascade import (
    _fine_score,
    coarse_rank,
    materialize_selected,
)
from ..feed_loop.tensor_policies import PERSONALIZED


class TensorPoiModelPolicy:
    eligible_fraction = 1.0
    observation_noise = PERSONALIZED.observation_noise
    local_observation_noise = PERSONALIZED.local_observation_noise
    realtime_interest_rate = PERSONALIZED.realtime_interest_rate
    multi_queue = False

    def __init__(
        self, bundle, stage: str, strength: float,
        coarse_keep: int = 20, fine_strength: float = 0.025,
        deployment_name: str | None = None,
    ):
        if stage not in {"coarse", "fine", "mix", "end_to_end"}:
            raise ValueError(f"unsupported POI model stage: {stage}")
        self.bundle = bundle
        self.stage = stage
        self.strength = strength
        self.fine_strength = fine_strength
        self.coarse_keep = coarse_keep
        self.name = deployment_name or (
            f"poi_{stage}_{bundle.name}_a{strength:.3f}"
        )

    def describe(self):
        return {
            "name": self.name,
            "model": self.bundle.name,
            "stage": self.stage,
            "strength": self.strength,
            "fine_strength": self.fine_strength,
            "coarse_keep": self.coarse_keep,
        }

    @staticmethod
    def _normalize(score):
        return (score - score.mean(dim=1, keepdim=True)) / (
            score.std(dim=1, keepdim=True).clamp_min(1e-4)
        )

    def select_candidate(self, user_ids, state, candidates, device, step, config):
        features = build_tensor_features(config, user_ids, state, candidates, step)
        shape = features.shape[:2]
        local = self.bundle.score(features.flatten(0, 1)).reshape(shape)
        local = self._normalize(local) * candidates["is_poi"]
        base, observed_affinity = _fine_score(
            PERSONALIZED, state["eligible"], user_ids, state, candidates
        )
        if self.stage in {"coarse", "end_to_end"}:
            coarse_scores = base + self.strength * local
            keep = min(self.coarse_keep, coarse_scores.shape[1])
            indices = torch.topk(coarse_scores, keep, dim=1).indices
            coarse_mask = torch.zeros_like(coarse_scores, dtype=torch.bool)
            coarse_mask.scatter_(1, indices, True)
            fine_scores = (
                base + (
                    self.fine_strength * local
                    if self.stage == "end_to_end" else 0.0
                )
            ).masked_fill(~coarse_mask, -1e9)
            mix_scores = fine_scores
        else:
            coarse_scores, coarse_mask, keep = coarse_rank(
                PERSONALIZED, observed_affinity, candidates, config.candidates
            )
            fine_strength = (
                self.strength if self.stage == "fine" else self.fine_strength
            )
            fine_scores = (
                base + fine_strength * local
            ).masked_fill(~coarse_mask, -1e9)
            mix_scores = fine_scores + (
                self.strength * local if self.stage == "mix" else 0.0
            )
        fine_choice = fine_scores.argmax(dim=1)
        choice = mix_scores.argmax(dim=1)
        return materialize_selected(
            self, user_ids, state, candidates, choice, fine_choice,
            fine_scores, mix_scores, coarse_scores, coarse_mask, keep, device,
        )
