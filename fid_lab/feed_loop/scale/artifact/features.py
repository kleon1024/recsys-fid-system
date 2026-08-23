"""GPU construction of the canonical 24-field stateful Feed feature vector."""

from __future__ import annotations

import torch

from ....simulation.environment import FEATURE_NAMES


def build_tensor_features(config, user_ids, state, candidates, step):
    observed_affinity = torch.einsum(
        "bkd,bd->bk", candidates["topics"], state["observed_interest"]
    )
    local_affinity = torch.einsum(
        "bkd,bd->bk", candidates["topics"], state["local_observed_interest"]
    )
    topic = candidates["candidate_topic"]
    short_match = (topic == state["last_topic"][:, None]).float()
    counts = state["topic_counts"]
    long_match = counts.gather(1, topic) / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    users, candidates_count = topic.shape
    repeated = lambda value: value[:, None].expand(users, candidates_count)
    same_city = candidates["same_city"]
    values = (
        observed_affinity,
        candidates["quality"],
        candidates["commerce"],
        candidates["popularity"],
        short_match,
        same_city,
        repeated(state["satisfaction"]),
        repeated(state["fatigue"]),
        repeated(state["trust"]),
        repeated(state["commerce_propensity"]),
        torch.full_like(observed_affinity, step / max(config.steps, 1)),
        long_match,
        torch.log1p(candidates["duration"]) / torch.log(
            torch.tensor(181.0, device=observed_affinity.device)
        ),
        candidates["is_poi"],
        repeated(torch.remainder(user_ids, 1024).float() / 1023.0),
        torch.remainder(candidates["item_ids"], 4096).float() / 4095.0,
        torch.remainder(candidates["author"], 1024).float() / 1023.0,
        topic.float() / max(config.topics - 1, 1),
        candidates["search_match"],
        candidates["retarget_match"],
        candidates["poi_quality"],
        candidates["inventory"],
        torch.where(same_city.bool(), torch.ones_like(same_city), torch.full_like(same_city, 0.05)),
        local_affinity,
    )
    features = torch.stack(values, dim=2).float()
    if features.shape[2] != len(FEATURE_NAMES):
        raise RuntimeError("tensor feature schema does not match stateful authority")
    if not torch.isfinite(features).all():
        raise RuntimeError("tensor features contain non-finite values")
    return features


def nonlinear_stay_adjustment(features: torch.Tensor) -> torch.Tensor:
    affinity = features[..., 0]
    quality = features[..., 1]
    short_match = features[..., 4]
    long_match = features[..., 11]
    user_segment = torch.remainder(torch.round(features[..., 14] * 1023), 7)
    item_segment = torch.remainder(torch.round(features[..., 15] * 4095), 7)
    segment_match = 1.0 - torch.abs(user_segment - item_segment) / 6.0
    threshold = ((affinity > 0.28) & (quality > 0.62)).float()
    sequence_novelty = short_match * (1.0 - long_match)
    periodic = torch.sin(torch.pi * features[..., 17] * (1.0 + features[..., 10]))
    return (
        0.75 * torch.tanh(2.4 * affinity * quality)
        + 0.50 * threshold
        + 0.35 * segment_match
        + 0.45 * sequence_novelty
        + 0.25 * periodic
        - 1.45
    )
