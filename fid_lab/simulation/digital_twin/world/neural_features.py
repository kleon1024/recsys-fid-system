"""Task-scoped v4 bridge into the existing 28-field NeuralSCM contract."""

from __future__ import annotations

import math

import torch

from ....feed_loop.world_model.contracts import (
    STOCHASTIC_ACTIONS,
    WorldModelConfig,
)
from ....feed_loop.world_model.ensemble import StructuralNoise
from ....feed_loop.world_model.feature_contract import feature_contract
from ....feed_loop.world_model.feature_contract import (
    CANONICAL_FEATURE_FIELDS,
    V4_REQUIRED_FEATURES,
)
from ...randomness.counter import (
    uniform_for_item_channels,
    uniform_for_items,
)
from ..catalog import PublicCatalog
from ..contracts import RenderedSlateBatch
from .state import UserWorldSnapshot
from .experience import candidate_experience


NEURAL_FEATURE_VERSION = "v5-experience-aware-feed-bridge-v1"
V4_FEATURE_CONTRACT = feature_contract(CANONICAL_FEATURE_FIELDS)
V4_FEATURE_COVERAGE = {
    str(index): (
        "native_v4" if index in V4_REQUIRED_FEATURES else "unused"
    )
    for index in range(len(CANONICAL_FEATURE_FIELDS))
}


def build_neural_scm_batch(
    snapshot: UserWorldSnapshot,
    catalog: PublicCatalog,
    slate: RenderedSlateBatch,
) -> dict[str, torch.Tensor]:
    users = snapshot.users
    row = slate.user_id
    item = slate.item_ids.clamp_min(0)
    history = users.behavior_sequence[row].float()
    history_present = history.abs().sum(dim=2) > 0
    denominator = history_present.float().sum(dim=1).clamp_min(1.0)
    topic_denominator = max(int(catalog.topic_id.max()), 1)
    topic = catalog.topic_id[item].float() / topic_denominator
    topic_match = (
        (history[:, None, :, 0] - topic[:, :, None]).abs()
        <= 0.5 / topic_denominator
    ) & history_present[:, None]
    recent_match = topic_match[:, :, -3:].any(dim=2).float()
    long_match = topic_match.float().sum(dim=2) / denominator[:, None]
    topic_affinity = (0.65 * recent_match + 0.35 * long_match).clamp(0.0, 1.0)
    engagement = history[:, :, (2, 3, 4, 6)].sum(dim=(1, 2)) / (
        4.0 * denominator
    )
    negative = history[:, :, 5].sum(dim=1) / denominator
    features = torch.zeros(
        *item.shape, 28, device=item.device, dtype=torch.float,
    )
    prior = catalog.quality_prior[item]
    features[:, :, 0] = topic_affinity
    features[:, :, 1] = prior.sqrt()
    features[:, :, 3] = prior
    features[:, :, 4] = recent_match
    features[:, :, 5] = (
        catalog.region[item] == users.region[row, None]
    ).float()
    features[:, :, 6] = engagement[:, None]
    features[:, :, 7] = negative[:, None]
    features[:, :, 8] = users.activity[row, None]
    features[:, :, 9] = users.habit[row, None]
    features[:, :, 10] = (denominator / history.shape[1])[:, None]
    features[:, :, 11] = long_match
    duration = catalog.duration_seconds[item].clamp(1.0, 180.0)
    features[:, :, 12] = torch.log1p(duration) / math.log(181.0)
    experience = candidate_experience(snapshot, catalog, slate)
    features[:, :, 13] = torch.log1p(experience.exact_repeat) / math.log(65.0)
    features[:, :, 14] = torch.log1p(
        experience.creator_pressure,
    ) / math.log(65.0)
    features[:, :, 15] = torch.log1p(
        experience.topic_pressure,
    ) / math.log(65.0)
    features[:, :, 16] = users.disappointment[row, None]
    features[:, :, 17] = topic
    features[:, :, 18] = (
        catalog.country[item] == users.country[row, None]
    ).float()
    features[:, :, 19] = recent_match
    features[:, :, 20] = 0.0
    features[:, :, 21] = 1.0
    age = (slate.event_time[:, None] - snapshot.item_publish_time[item]).clamp_min(0)
    features[:, :, 22] = torch.log1p(age.float()) / math.log1p(
        30 * snapshot.ticks_per_day
    )
    features[:, :, 23] = users.novelty[row, None]
    account_age_days = (
        (slate.event_time - users.signup_time[row]).clamp_min(0).float()
        / snapshot.ticks_per_day
    )
    features[:, :, 24] = (
        torch.log1p(account_age_days) / math.log1p(4_000.0)
    )[:, None]
    features[:, :, 25] = users.activity[row, None]
    features[:, :, 26] = users.lifecycle_cohort[row, None].float() / 3.0
    features[:, :, 27] = experience.repeat_penalty.clamp_max(4.0) / 4.0
    features.masked_fill_(~slate.valid[:, :, None], 0.0)
    return {
        "slate_features": features,
        "sequence": history,
        "lifecycle": users.lifecycle_cohort[row].long(),
        "region": torch.remainder(users.region[row], 10).long(),
    }


def _normal_channels(
    slate: RenderedSlateBatch, channels: int, stream: int, seed: int,
) -> torch.Tensor:
    channel = torch.arange(channels, device=slate.item_ids.device)[None, None]
    item = (
        slate.item_ids.clamp_min(0) * 131
        + slate.positions.clamp_min(0) * 17
    )[:, :, None].expand(-1, -1, channels)
    first = uniform_for_item_channels(
        slate.request_id,
        item,
        channel.expand_as(item),
        0,
        stream,
        seed,
    ).clamp_min(1e-7)
    second = uniform_for_item_channels(
        slate.request_id,
        item,
        channel.expand_as(item),
        0,
        stream + 1,
        seed,
    )
    return torch.sqrt(-2.0 * torch.log(first)) * torch.cos(
        2.0 * torch.pi * second
    )


def request_keyed_structural_noise(
    slate: RenderedSlateBatch,
    config: WorldModelConfig,
    seed: int,
) -> StructuralNoise:
    item_key = (
        slate.item_ids.clamp_min(0) * 131
        + slate.positions.clamp_min(0) * 17
    )
    return StructuralNoise(
        latent=_normal_channels(slate, config.latent_dim, 1_701, seed),
        mixture=uniform_for_items(
            slate.request_id, item_key, 0, 1_709, seed,
        ),
        stay=_normal_channels(slate, 1, 1_717, seed).squeeze(2),
        actions=uniform_for_item_channels(
            slate.request_id,
            item_key[:, :, None].expand(-1, -1, len(STOCHASTIC_ACTIONS)),
            torch.arange(
                len(STOCHASTIC_ACTIONS), device=item_key.device,
            )[None, None].expand(-1, item_key.shape[1], -1),
            0,
            1_727,
            seed,
        ),
    )
