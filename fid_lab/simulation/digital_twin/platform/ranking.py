"""Observable coarse, fine and diversity stages for the reference platform."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..catalog import PublicCatalog
from ..contracts import EventType, PlatformRequestBatch
from .projection import ITEM_COUNTER_EVENTS, PlatformProjectionState
from .retrieval import ROUTE_NAMES, MultiRouteRetriever, RetrievalResult


@dataclass(frozen=True)
class RankingConfig:
    coarse_k: int = 64
    fine_k: int = 24
    expose_k: int = 8
    creator_penalty: float = 0.18
    kind_penalty: float = 0.08

    def __post_init__(self):
        if not self.coarse_k >= self.fine_k >= self.expose_k > 0:
            raise ValueError("ranking budgets must satisfy coarse >= fine >= expose")


@dataclass(frozen=True)
class CascadePolicy:
    name: str
    coarse_version_id: int
    fine_version_id: int
    mix_version_id: int
    recall_version_id: int = 1
    enabled_routes: tuple[str, ...] = ROUTE_NAMES
    coarse_weights: tuple[float, ...] = (
        0.42, 0.16, 0.08, 0.06, 0.08, 0.07, 0.04, 0.04, 0.05, -0.06,
    )
    fine_weights: tuple[float, ...] = (
        0.48, 0.15, 0.06, 0.06, 0.06, 0.06, 0.03, 0.03, 0.06, -0.08,
    )
    cross_weight: float = 0.12
    sequence_weight: float = 0.18

    def __post_init__(self):
        if len(self.coarse_weights) != 10 or len(self.fine_weights) != 10:
            raise ValueError("reference ranker expects ten observable features")
        unknown = set(self.enabled_routes) - set(ROUTE_NAMES)
        if unknown:
            raise ValueError(f"unknown retrieval routes: {sorted(unknown)}")
        if not self.enabled_routes:
            raise ValueError("at least one retrieval route must be enabled")


@dataclass(frozen=True)
class RankedStages:
    coarse_item_id: torch.Tensor
    coarse_score: torch.Tensor
    fine_item_id: torch.Tensor
    fine_score: torch.Tensor
    exposed_item_id: torch.Tensor
    exposed_score: torch.Tensor
    exposed_position: torch.Tensor


class CascadeRanker:
    def __init__(
        self,
        catalog: PublicCatalog,
        retriever: MultiRouteRetriever,
        config: RankingConfig,
    ):
        self.catalog = catalog
        self.retriever = retriever
        self.config = config

    def _features(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
        recall_score: torch.Tensor,
    ) -> torch.Tensor:
        item = item_id.clamp_min(0)
        valid = item_id >= 0
        query = self.retriever.query_embedding(requests, state)
        affinity = torch.einsum(
            "bkd,bd->bk", self.catalog.content_embedding[item], query,
        )
        impression = state.item_event_counts[
            item, ITEM_COUNTER_EVENTS.index(EventType.IMPRESSION)
        ]
        engagement = (
            state.item_event_counts[
                item, ITEM_COUNTER_EVENTS.index(EventType.LONG_VIEW)
            ]
            + state.item_event_counts[
                item, ITEM_COUNTER_EVENTS.index(EventType.CLICK)
            ]
        )
        engagement_rate = engagement / impression.clamp_min(1.0)
        age = (
            requests.event_time[:, None]
            - state.item_publish_time[item].clamp_max(
                requests.event_time[:, None]
            )
        ).clamp_min(0).float()
        freshness = torch.exp(-age / 192.0)
        country = state.user_country[requests.user_id]
        region = state.user_region[requests.user_id]
        geo = (
            0.35 * (self.catalog.country[item] == country[:, None]).float()
            + 0.65 * (self.catalog.region[item] == region[:, None]).float()
        )
        creator = self.catalog.creator_id[item]
        creator_rate = state.creator_engagements[creator] / (
            state.creator_impressions[creator].clamp_min(1.0)
        )
        history = state.user_history_item[requests.user_id]
        repeated = (item[:, :, None] == history[:, None, :]).any(dim=2).float()
        features = torch.stack((
            affinity,
            self.catalog.quality_prior[item],
            torch.log1p(impression) / 8.0,
            engagement_rate.clamp_max(1.0),
            freshness,
            geo,
            state.item_inventory[item],
            recall_score.clamp_min(0.0) * 20.0,
            creator_rate.clamp_max(1.0),
            repeated,
        ), dim=2)
        return features.masked_fill(~valid[:, :, None], 0.0)

    def _sequence_score(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
    ) -> torch.Tensor:
        item = item_id.clamp_min(0)
        history = state.user_history_item[requests.user_id]
        history_valid = history >= 0
        candidate = self.catalog.content_embedding[item]
        history_embedding = self.catalog.content_embedding[history.clamp_min(0)]
        similarity = torch.einsum(
            "bkd,bhd->bkh", candidate, history_embedding,
        ).masked_fill(~history_valid[:, None, :], -20.0)
        attention = torch.softmax(similarity, dim=2)
        attention = attention * history_valid[:, None, :].float()
        attention /= attention.sum(dim=2, keepdim=True).clamp_min(1e-8)
        interest = torch.einsum(
            "bkh,bhd->bkd", attention, history_embedding,
        )
        return torch.einsum("bkd,bkd->bk", candidate, interest)

    @staticmethod
    def _top(
        item: torch.Tensor, score: torch.Tensor, limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = min(limit, item.shape[1])
        valid_score = score.masked_fill(item < 0, -torch.inf)
        position = torch.topk(valid_score, width, dim=1).indices
        selected_item = torch.gather(item, 1, position)
        selected_score = torch.gather(valid_score, 1, position)
        selected_item = torch.where(
            torch.isfinite(selected_score),
            selected_item,
            torch.full_like(selected_item, -1),
        )
        if width < limit:
            padding = limit - width
            selected_item = torch.nn.functional.pad(
                selected_item, (0, padding), value=-1,
            )
            selected_score = torch.nn.functional.pad(
                selected_score, (0, padding), value=-torch.inf,
            )
        return selected_item, selected_score

    def rank(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        retrieval: RetrievalResult,
        policy: CascadePolicy,
    ) -> RankedStages:
        coarse_features = self._features(
            requests, state, retrieval.item_id, retrieval.score,
        )
        coarse_weight = torch.tensor(
            policy.coarse_weights,
            device=retrieval.item_id.device,
        )
        coarse_raw = torch.einsum("bkd,d->bk", coarse_features, coarse_weight)
        coarse_raw += policy.cross_weight * (
            coarse_features[:, :, 0] * coarse_features[:, :, 1]
            + coarse_features[:, :, 4] * coarse_features[:, :, 5]
        )
        coarse_item, coarse_score = self._top(
            retrieval.item_id, coarse_raw, self.config.coarse_k,
        )
        coarse_recall_score = self._map_score(
            coarse_item, retrieval.item_id, retrieval.score,
        )
        fine_features = self._features(
            requests, state, coarse_item, coarse_recall_score,
        )
        fine_weight = torch.tensor(
            policy.fine_weights, device=retrieval.item_id.device,
        )
        fine_raw = torch.einsum("bkd,d->bk", fine_features, fine_weight)
        fine_raw += policy.cross_weight * torch.sin(
            1.7 * fine_features[:, :, 0]
        ) * fine_features[:, :, 1]
        fine_raw += policy.sequence_weight * self._sequence_score(
            requests, state, coarse_item,
        )
        fine_item, fine_score = self._top(
            coarse_item, fine_raw, self.config.fine_k,
        )
        exposed_item, exposed_score = self._diversified_top(
            fine_item, fine_score,
        )
        position = torch.arange(
            self.config.expose_k, device=fine_item.device,
        )[None].expand(len(fine_item), -1)
        return RankedStages(
            coarse_item,
            coarse_score,
            fine_item,
            fine_score,
            exposed_item,
            exposed_score,
            position,
        )

    @staticmethod
    def _map_score(
        child: torch.Tensor, parent: torch.Tensor, score: torch.Tensor,
    ) -> torch.Tensor:
        match = child[:, :, None] == parent[:, None, :]
        location = match.float().argmax(dim=2)
        mapped = torch.gather(score, 1, location)
        return torch.where(match.any(dim=2), mapped, torch.zeros_like(mapped))

    def _diversified_top(
        self, item_id: torch.Tensor, score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requests = len(item_id)
        selected_item = torch.full(
            (requests, self.config.expose_k),
            -1,
            device=item_id.device,
            dtype=torch.long,
        )
        selected_score = torch.full(
            (requests, self.config.expose_k),
            -torch.inf,
            device=item_id.device,
        )
        available = item_id >= 0
        safe_item = item_id.clamp_min(0)
        creator = self.catalog.creator_id[safe_item]
        kind = self.catalog.content_kind[safe_item]
        rows = torch.arange(requests, device=item_id.device)
        for position in range(self.config.expose_k):
            prior_item = selected_item[:, :position].clamp_min(0)
            prior_valid = selected_item[:, :position] >= 0
            creator_count = (
                creator[:, :, None]
                == self.catalog.creator_id[prior_item][:, None, :]
            ) & prior_valid[:, None, :]
            kind_count = (
                kind[:, :, None]
                == self.catalog.content_kind[prior_item][:, None, :]
            ) & prior_valid[:, None, :]
            adjusted = (
                score
                - self.config.creator_penalty * creator_count.sum(dim=2)
                - self.config.kind_penalty * kind_count.sum(dim=2)
            ).masked_fill(~available, -torch.inf)
            choice = adjusted.argmax(dim=1)
            selected_item[:, position] = item_id[rows, choice]
            selected_score[:, position] = score[rows, choice]
            available[rows, choice] = False
        return selected_item, selected_score
