"""Observable pair features and baseline score for Feed posting."""

from __future__ import annotations

import torch


FEATURE_NAMES = (
    "profile_similarity", "sequence_similarity", "creator_similarity",
    "same_recent_category", "trend", "quality", "difficulty_match",
    "saturation", "experience", "fatigue", "activity", "country_match",
    "route_history", "route_semantic",
)


def candidate_features(world, candidates):
    catalog, requests = world.catalog, world.requests
    prompt_ids = candidates.prompt_ids
    profile = torch.einsum(
        "bkd,bd->bk", catalog.semantic[prompt_ids], requests.observed_profile
    )
    sequence = torch.einsum(
        "bkd,bd->bk", catalog.semantic[prompt_ids], requests.sequence_summary
    )
    creator = torch.einsum(
        "bkd,bd->bk", catalog.semantic[prompt_ids], requests.creator_history
    )
    difficulty = 1.0 - (
        catalog.difficulty[prompt_ids] - requests.experience[:, None]
    ).abs()
    return torch.stack((
        profile, sequence, creator,
        (catalog.category[prompt_ids] == requests.recent_category[:, None]).float(),
        catalog.trend[prompt_ids], catalog.quality[prompt_ids], difficulty,
        catalog.saturation[prompt_ids],
        requests.experience[:, None].expand_as(profile),
        requests.fatigue[:, None].expand_as(profile),
        requests.activity[:, None].expand_as(profile),
        (catalog.country[prompt_ids] == requests.country[:, None]).float(),
        ((candidates.route_bits & (1 << 2)) > 0).float(),
        ((candidates.route_bits & (1 << 3)) > 0).float(),
    ), dim=2)


def rule_score(features):
    return (
        0.45 * features[:, :, 4]
        + 0.30 * features[:, :, 3]
        + 0.20 * features[:, :, 5]
        - 0.20 * features[:, :, 7]
    )
