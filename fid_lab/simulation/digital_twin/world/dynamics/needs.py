"""Heterogeneous need episodes that evolve independently of platform features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from ....randomness.counter import normal, uniform


NEED_DYNAMICS_VERSION = "heterogeneous-need-episodes-v1"


class NeedKind(IntEnum):
    ENTERTAINMENT = 0
    INFORMATION = 1
    SOCIAL = 2
    LOCAL = 3
    COMMERCE = 4
    CREATION = 5


@dataclass(frozen=True)
class NeedPopulation:
    kind: torch.Tensor
    topic: torch.Tensor
    strength: torch.Tensor
    expiry_time: torch.Tensor


def sample_need_population(
    user: torch.Tensor,
    primary_topic: torch.Tensor,
    secondary_topic: torch.Tensor,
    surface_intent: torch.Tensor,
    ticks_per_day: int,
    topics: int,
    seed: int,
) -> NeedPopulation:
    logits = torch.log(surface_intent.clamp_min(1e-6))
    kind_logits = torch.stack((
        logits[:, 0],
        logits[:, 1],
        0.45 * logits[:, 0] + 0.30,
        logits[:, 2],
        logits[:, 3],
        logits[:, 5],
    ), dim=1)
    draw = uniform(user, 0, 1_701, seed, len(NeedKind)).clamp(1e-6, 1.0 - 1e-6)
    kind = torch.argmax(kind_logits - torch.log(-torch.log(draw)), dim=1)
    topic_draw = uniform(user, 0, 1_703, seed)
    random_topic = torch.floor(
        topics * uniform(user, 0, 1_705, seed)
    ).long().clamp_max(topics - 1)
    topic = torch.where(
        topic_draw < 0.62,
        primary_topic,
        torch.where(topic_draw < 0.88, secondary_topic, random_topic),
    )
    strength = torch.sigmoid(
        0.35 + 0.75 * normal(user, 0, 1_707, seed)
    )
    lifetime = 1 + torch.floor(
        ticks_per_day
        * (0.15 + 5.0 * uniform(user, 0, 1_709, seed).square())
    ).long()
    return NeedPopulation(kind, topic, strength, lifetime)


def refresh_expired_needs(
    *,
    user: torch.Tensor,
    need_kind: torch.Tensor,
    need_topic: torch.Tensor,
    need_strength: torch.Tensor,
    need_expiry_time: torch.Tensor,
    primary_topic: torch.Tensor,
    secondary_topic: torch.Tensor,
    logical_time: int,
    ticks_per_day: int,
    topics: int,
    seed: int,
) -> None:
    expired = need_expiry_time <= logical_time
    if not expired.any():
        return
    selected = user[expired]
    draw = uniform(selected, logical_time, 1_719, seed)
    next_kind = torch.floor(
        len(NeedKind) * uniform(selected, logical_time, 1_721, seed)
    ).long().clamp_max(len(NeedKind) - 1)
    random_topic = torch.floor(
        topics * uniform(selected, logical_time, 1_723, seed)
    ).long().clamp_max(topics - 1)
    next_topic = torch.where(
        draw < 0.58,
        primary_topic[expired],
        torch.where(draw < 0.84, secondary_topic[expired], random_topic),
    )
    next_strength = torch.sigmoid(
        0.20 + 0.85 * normal(selected, logical_time, 1_727, seed)
    )
    lifetime = 1 + torch.floor(
        ticks_per_day
        * (0.10 + 6.0 * uniform(selected, logical_time, 1_729, seed).square())
    ).long()
    need_kind[expired] = next_kind
    need_topic[expired] = next_topic
    need_strength[expired] = next_strength
    need_expiry_time[expired] = logical_time + lifetime
