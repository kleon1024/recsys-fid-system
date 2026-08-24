"""Exogenous shocks and factual recommendation-induced topic trends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....randomness.counter import normal
from ...contracts import AppEventBatch, EventType


TREND_VERSION = "regional-topic-ar-feedback-v1"


@dataclass
class TrendState:
    strength: torch.Tensor
    momentum: torch.Tensor
    last_time: int


class TrendProcess:
    def __init__(self, regions: int, topics: int, seed: int, device):
        self.regions = regions
        self.topics = topics
        self.seed = seed
        self._entity = torch.arange(regions * topics, device=device)
        self.state = TrendState(
            strength=torch.zeros(regions, topics, device=device),
            momentum=torch.zeros(regions, topics, device=device),
            last_time=-1,
        )

    def advance(self, logical_time: int) -> None:
        if logical_time <= self.state.last_time:
            return
        gap = logical_time - self.state.last_time
        decay = 0.985 ** gap
        innovation = normal(
            self._entity,
            logical_time,
            1_601,
            self.seed,
        ).reshape(self.regions, self.topics)
        common_topic = normal(
            torch.arange(self.topics, device=self._entity.device),
            logical_time,
            1_607,
            self.seed,
        )[None]
        self.state.momentum = (
            decay * self.state.momentum
            + 0.035 * innovation
            + 0.018 * common_topic
        ).clamp(-1.5, 1.5)
        self.state.strength = (
            decay * self.state.strength + self.state.momentum
        ).clamp(-2.5, 2.5)
        self.state.last_time = logical_time

    def commit(self, events: AppEventBatch) -> None:
        if not len(events.event_id):
            return
        weight = torch.zeros(len(events.event_id), device=events.event_id.device)
        positive = {
            EventType.LONG_VIEW: 0.35,
            EventType.COMPLETE: 0.45,
            EventType.LIKE: 0.30,
            EventType.COMMENT: 0.38,
            EventType.SHARE: 0.55,
            EventType.FOLLOW: 0.50,
            EventType.CLICK: 0.18,
            EventType.FAVORITE: 0.42,
            EventType.ORDER: 0.65,
            EventType.PAYMENT: 0.85,
            EventType.PUBLISH: 0.55,
        }
        for event_type, value in positive.items():
            weight[events.event(event_type)] = value
        weight[events.event(EventType.NEGATIVE)] = -0.55
        valid = (
            (weight != 0.0)
            & (events.region >= 0)
            & (events.region < self.regions)
            & (events.topic_id >= 0)
            & (events.topic_id < self.topics)
        )
        if not valid.any():
            return
        index = events.region[valid] * self.topics + events.topic_id[valid]
        aggregate = torch.zeros(
            self.regions * self.topics, device=events.event_id.device,
        )
        count = torch.zeros_like(aggregate)
        aggregate.index_add_(0, index, weight[valid])
        count.index_add_(0, index, torch.ones_like(weight[valid]))
        response = (aggregate / count.clamp_min(1.0)).reshape(
            self.regions, self.topics,
        )
        self.state.momentum = (
            self.state.momentum + 0.025 * response
        ).clamp(-1.5, 1.5)

    def snapshot(self) -> torch.Tensor:
        return self.state.strength.clone()

    def top_topic(self, region: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.state.strength[region], dim=1)
