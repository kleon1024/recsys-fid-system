"""Observable coarse, fine and diversity stages for the reference platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..catalog import PublicCatalog
from ..contracts import PlatformRequestBatch
from .features.encoder import FeatureTensorBatch, PlatformFeatureEncoder
from .features.manifest import FeatureManifest
from .projection import PlatformProjectionState
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
    fine_features: FeatureTensorBatch


class LearnedFineScorer(Protocol):
    feature_manifest_hash: str

    def score(
        self,
        features: FeatureTensorBatch,
        surface: torch.Tensor,
    ) -> torch.Tensor: ...


class CascadeRanker:
    def __init__(
        self,
        catalog: PublicCatalog,
        retriever: MultiRouteRetriever,
        config: RankingConfig,
        ticks_per_day: int,
        feature_manifest: FeatureManifest,
    ):
        self.catalog = catalog
        self.retriever = retriever
        self.config = config
        self.features = PlatformFeatureEncoder(
            catalog, retriever, ticks_per_day, feature_manifest,
        )
        self._fine_scorers: dict[int, LearnedFineScorer] = {}

    def install_fine_scorer(
        self, serving_version_id: int, scorer: LearnedFineScorer,
    ) -> None:
        if serving_version_id <= 0:
            raise ValueError("learned serving version must be positive")
        if scorer.feature_manifest_hash != self.features.manifest.manifest_hash:
            raise ValueError("fine scorer feature manifest is incompatible")
        self._fine_scorers[serving_version_id] = scorer

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
        coarse_features = self.features.base_dense(
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
        coarse_route_bits = self._map_score(
            coarse_item, retrieval.item_id, retrieval.route_bits,
        )
        coarse_feature_tensors = self.features.encode(
            requests,
            state,
            coarse_item,
            coarse_recall_score,
            coarse_route_bits,
        )
        scorer = self._fine_scorers.get(policy.fine_version_id)
        if scorer is None:
            fine_features = coarse_feature_tensors.dense[:, :, :10]
            fine_weight = torch.tensor(
                policy.fine_weights, device=retrieval.item_id.device,
            )
            fine_raw = torch.einsum("bkd,d->bk", fine_features, fine_weight)
            fine_raw += policy.cross_weight * torch.sin(
                1.7 * fine_features[:, :, 0]
            ) * fine_features[:, :, 1]
            fine_raw += policy.sequence_weight * (
                coarse_feature_tensors.dense[:, :, 10]
            )
        else:
            fine_raw = scorer.score(coarse_feature_tensors, requests.surface)
            if fine_raw.shape != coarse_item.shape:
                raise ValueError("learned fine scorer returned an invalid shape")
            fine_raw = fine_raw.to(coarse_item.device)
        fine_raw = fine_raw.masked_fill(coarse_item < 0, -torch.inf)
        fine_item, fine_score = self._top(
            coarse_item, fine_raw, self.config.fine_k,
        )
        fine_recall_score = self._map_score(
            fine_item, retrieval.item_id, retrieval.score,
        )
        fine_route_bits = self._map_score(
            fine_item, retrieval.item_id, retrieval.route_bits,
        )
        feature_tensors = self.features.encode(
            requests, state, fine_item, fine_recall_score, fine_route_bits,
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
            feature_tensors,
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
            choice_is_valid = torch.isfinite(adjusted[rows, choice])
            selected_item[:, position] = torch.where(
                choice_is_valid,
                item_id[rows, choice],
                torch.full_like(choice, -1),
            )
            selected_score[:, position] = torch.where(
                choice_is_valid,
                score[rows, choice],
                torch.full_like(score[rows, choice], -torch.inf),
            )
            valid_rows = rows[choice_is_valid]
            available[valid_rows, choice[choice_is_valid]] = False
        return selected_item, selected_score
