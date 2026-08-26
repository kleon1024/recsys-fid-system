"""Observable coarse, fine and diversity stages for the reference platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..catalog import PublicCatalog
from ..contracts import ContentKind, PlatformRequestBatch, SelectionPolicyKind
from .exploration import (
    exploration_mask,
    mixture_admission_probability,
    mixture_position_probability,
    mixture_slate_log_probability,
    random_ordered_top,
)
from .routes.cold_start_exploration import (
    ColdStartDraw,
    draw_cold_start_item,
    inject_last,
    targeted_admission_probability,
    targeted_position_probability,
    targeted_slate_log_probability,
)
from .features.encoder import FeatureTensorBatch, PlatformFeatureEncoder
from .features.manifest import FeatureManifest
from .projection import PlatformProjectionState
from .retrieval import ROUTE_NAMES, MultiRouteRetriever, RetrievalResult
from .routes import (
    BUSINESS_ROUTE_NAMES,
    DEFAULT_BUSINESS_ROUTE_NAMES,
    FEED_ROUTE_NAMES,
)


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
    enabled_routes: tuple[str, ...] = FEED_ROUTE_NAMES
    enabled_business_routes: tuple[str, ...] = DEFAULT_BUSINESS_ROUTE_NAMES
    coarse_weights: tuple[float, ...] = (
        0.42, 0.16, 0.08, 0.06, 0.08, 0.07, 0.04, 0.04, 0.05, -0.06,
    )
    fine_weights: tuple[float, ...] = (
        0.48, 0.15, 0.06, 0.06, 0.06, 0.06, 0.03, 0.03, 0.06, -0.08,
    )
    cross_weight: float = 0.12
    sequence_weight: float = 0.18
    exploration_rate: float = 0.0
    exploration_seed: int = 1_991
    feed_exposure_dedup_ticks: int = 0
    feed_session_dedup: bool = False
    cold_start_exploration_rate: float = 0.0
    commerce_require_inventory: bool = False
    commerce_min_inventory: float = 0.0

    def __post_init__(self):
        if len(self.coarse_weights) != 10 or len(self.fine_weights) != 10:
            raise ValueError("reference ranker expects ten observable features")
        unknown = set(self.enabled_routes) - set(ROUTE_NAMES)
        if unknown:
            raise ValueError(f"unknown retrieval routes: {sorted(unknown)}")
        if not self.enabled_routes:
            raise ValueError("at least one retrieval route must be enabled")
        unknown_business = set(self.enabled_business_routes) - set(
            BUSINESS_ROUTE_NAMES
        )
        if unknown_business:
            raise ValueError(
                f"unknown business retrieval routes: {sorted(unknown_business)}"
            )
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration rate must be in [0, 1]")
        if self.feed_exposure_dedup_ticks < 0:
            raise ValueError("Feed exposure dedup window cannot be negative")
        if not 0.0 <= self.cold_start_exploration_rate <= 1.0:
            raise ValueError("cold-start exploration rate must be in [0, 1]")
        if self.exploration_rate and self.cold_start_exploration_rate:
            raise ValueError("general and cold-start exploration cannot overlap")
        if self.cold_start_exploration_rate and "cold_start" not in self.enabled_routes:
            raise ValueError("cold-start exploration requires its recall route")
        if not 0.0 <= self.commerce_min_inventory <= 1.0:
            raise ValueError("Commerce minimum inventory must be in [0, 1]")

    @property
    def effective_routes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((
            *self.enabled_routes,
            *self.enabled_business_routes,
        )))


@dataclass(frozen=True)
class RankedStages:
    coarse_input_score: torch.Tensor
    coarse_admission_probability: torch.Tensor
    coarse_item_id: torch.Tensor
    coarse_selected_score: torch.Tensor
    fine_input_score: torch.Tensor
    fine_admission_probability: torch.Tensor
    fine_item_id: torch.Tensor
    fine_selected_score: torch.Tensor
    exposed_item_id: torch.Tensor
    exposed_score: torch.Tensor
    exposed_position: torch.Tensor
    exposure_probability: torch.Tensor
    candidate_exposure_probability: torch.Tensor
    selection_policy_kind: torch.Tensor
    exploration_rate: torch.Tensor
    slate_log_probability: torch.Tensor
    candidate_features: FeatureTensorBatch


class LearnedRankScorer(Protocol):
    feature_manifest_hash: str

    def score(
        self,
        features: FeatureTensorBatch,
        surface: torch.Tensor,
    ) -> torch.Tensor: ...


LearnedCoarseScorer = LearnedRankScorer
LearnedFineScorer = LearnedRankScorer


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
        self._coarse_scorers: dict[int, LearnedCoarseScorer] = {}
        self._fine_scorers: dict[int, LearnedFineScorer] = {}

    def install_coarse_scorer(
        self, serving_version_id: int, scorer: LearnedCoarseScorer,
    ) -> None:
        self._install_scorer(
            self._coarse_scorers, serving_version_id, scorer, "coarse",
        )

    def install_fine_scorer(
        self, serving_version_id: int, scorer: LearnedFineScorer,
    ) -> None:
        self._install_scorer(
            self._fine_scorers, serving_version_id, scorer, "fine",
        )

    def _install_scorer(
        self,
        registry: dict[int, LearnedRankScorer],
        serving_version_id: int,
        scorer: LearnedRankScorer,
        stage: str,
    ) -> None:
        if serving_version_id <= 0:
            raise ValueError("learned serving version must be positive")
        if scorer.feature_manifest_hash != self.features.manifest.manifest_hash:
            raise ValueError(f"{stage} scorer feature manifest is incompatible")
        registry[serving_version_id] = scorer

    def validate_policy_artifacts(self, policy: CascadePolicy) -> None:
        missing = []
        if policy.coarse_version_id > 0 and (
            policy.coarse_version_id not in self._coarse_scorers
        ):
            missing.append(f"coarse={policy.coarse_version_id}")
        if policy.fine_version_id > 0 and (
            policy.fine_version_id not in self._fine_scorers
        ):
            missing.append(f"fine={policy.fine_version_id}")
        if missing:
            raise ValueError(
                "policy references uninstalled model artifacts: "
                + ", ".join(missing)
            )

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
        candidate_features = self.features.encode(
            requests,
            state,
            retrieval.item_id,
            retrieval.score,
            retrieval.route_bits,
        )
        coarse_raw = self._coarse_scores(
            candidate_features, retrieval.item_id, requests, policy,
        )
        deterministic_coarse, _ = self._top(
            retrieval.item_id, coarse_raw, self.config.coarse_k,
        )
        random_coarse = random_ordered_top(
            requests.request_id,
            retrieval.item_id,
            self.config.coarse_k,
            policy.exploration_seed,
        )
        randomized = exploration_mask(
            requests.request_id,
            requests.event_time,
            policy.exploration_rate,
            policy.exploration_seed,
        )
        deterministic_input = self._select_features(
            deterministic_coarse, retrieval.item_id, candidate_features,
        )
        deterministic_fine_raw = self._fine_scores(
            deterministic_input, deterministic_coarse, requests, policy,
        )
        random_input = self._select_features(
            random_coarse, retrieval.item_id, candidate_features,
        )
        random_fine_raw = self._fine_scores(
            random_input, random_coarse, requests, policy,
        )
        coarse_item = torch.where(randomized[:, None], random_coarse, deterministic_coarse)
        coarse_selected_score = self._map_score(
            coarse_item, retrieval.item_id, coarse_raw,
        )
        fine_raw = torch.where(
            randomized[:, None], random_fine_raw, deterministic_fine_raw,
        )
        deterministic_fine, _ = self._top(
            deterministic_coarse, deterministic_fine_raw, self.config.fine_k,
        )
        random_fine = self._prefix(random_coarse, self.config.fine_k)
        fine_item = torch.where(
            randomized[:, None], random_fine, deterministic_fine,
        )
        fine_selected_score = self._map_score(fine_item, coarse_item, fine_raw)
        deterministic_exposed, deterministic_exposed_score = self._diversified_top(
            deterministic_fine,
            self._map_score(
                deterministic_fine, deterministic_coarse, deterministic_fine_raw,
            ),
        )
        if policy.cold_start_exploration_rate:
            return self._rank_with_cold_start_exploration(
                requests=requests, state=state, retrieval=retrieval, policy=policy,
                candidate_features=candidate_features,
                coarse_raw=coarse_raw, deterministic_coarse=deterministic_coarse,
                deterministic_fine=deterministic_fine,
                deterministic_exposed=deterministic_exposed,
            )
        random_exposed = self._randomized_exposed(random_fine)
        random_exposed_score = self._map_score(
            random_exposed, coarse_item, fine_raw,
        )
        exposed_item = torch.where(
            randomized[:, None], random_exposed, deterministic_exposed,
        )
        exposed_score = torch.where(
            randomized[:, None], random_exposed_score, deterministic_exposed_score,
        )
        position = torch.arange(
            self.config.expose_k, device=fine_item.device,
        )[None].expand(len(fine_item), -1)
        selection = self._selection_metadata(
            requests,
            retrieval,
            deterministic_coarse,
            deterministic_fine,
            deterministic_exposed,
            exposed_item,
            randomized,
            policy,
        )
        return RankedStages(
            coarse_raw,
            selection[0],
            coarse_item,
            coarse_selected_score,
            fine_raw,
            selection[1],
            fine_item,
            fine_selected_score,
            exposed_item,
            exposed_score,
            position,
            *selection[2:],
            candidate_features,
        )

    def _randomized_exposed(self, random_fine: torch.Tensor) -> torch.Tensor:
        priority = -torch.arange(
            random_fine.shape[1],
            device=random_fine.device,
            dtype=torch.float,
        )[None].expand(len(random_fine), -1)
        exposed, _ = self._diversified_top(random_fine, priority)
        return exposed

    def _rank_with_cold_start_exploration(
        self,
        *,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        retrieval: RetrievalResult,
        policy: CascadePolicy,
        candidate_features: FeatureTensorBatch,
        coarse_raw: torch.Tensor,
        deterministic_coarse: torch.Tensor,
        deterministic_fine: torch.Tensor,
        deterministic_exposed: torch.Tensor,
    ) -> RankedStages:
        draw = draw_cold_start_item(
            requests,
            state,
            self.catalog.content_kind,
            retrieval.item_id,
            retrieval.route_bits,
            deterministic_exposed,
            rate=policy.cold_start_exploration_rate,
            seed=policy.exploration_seed,
            cold_route_bit=1 << ROUTE_NAMES.index("cold_start"),
        )
        coarse_item = inject_last(
            deterministic_coarse, draw.item, draw.randomized,
        )
        coarse_selected_score = self._map_score(
            coarse_item, retrieval.item_id, coarse_raw,
        )
        fine_input = self._select_features(
            coarse_item, retrieval.item_id, candidate_features,
        )
        fine_raw = self._fine_scores(
            fine_input, coarse_item, requests, policy,
        )
        ranked_fine, _ = self._top(
            coarse_item, fine_raw, self.config.fine_k,
        )
        fine_item = inject_last(ranked_fine, draw.item, draw.randomized)
        fine_selected_score = self._map_score(
            fine_item, coarse_item, fine_raw,
        )
        ranked_exposed, _ = self._diversified_top(
            fine_item, fine_selected_score,
        )
        exposed_item = inject_last(
            ranked_exposed, draw.item, draw.randomized,
        )
        exposed_score = self._map_score(
            exposed_item, fine_item, fine_selected_score,
        )
        selection = self._cold_start_selection_metadata(
            requests,
            retrieval,
            deterministic_coarse,
            deterministic_fine,
            deterministic_exposed,
            exposed_item,
            draw,
            policy,
        )
        position = torch.arange(
            self.config.expose_k, device=fine_item.device,
        )[None].expand(len(fine_item), -1)
        return RankedStages(
            coarse_raw,
            selection[0],
            coarse_item,
            coarse_selected_score,
            fine_raw,
            selection[1],
            fine_item,
            fine_selected_score,
            exposed_item,
            exposed_score,
            position,
            *selection[2:],
            candidate_features,
        )

    @staticmethod
    def _cold_start_selection_metadata(
        requests: PlatformRequestBatch,
        retrieval: RetrievalResult,
        deterministic_coarse: torch.Tensor,
        deterministic_fine: torch.Tensor,
        deterministic_exposed: torch.Tensor,
        exposed_item: torch.Tensor,
        draw: ColdStartDraw,
        policy: CascadePolicy,
    ) -> tuple[torch.Tensor, ...]:
        rate = policy.cold_start_exploration_rate
        selection_kind = torch.where(
            draw.randomized,
            torch.full_like(
                requests.request_id, int(SelectionPolicyKind.RANDOMIZED),
            ),
            torch.full_like(
                requests.request_id, int(SelectionPolicyKind.DETERMINISTIC),
            ),
        )
        return (
            targeted_admission_probability(
                retrieval.item_id, deterministic_coarse, draw, rate,
            ),
            targeted_admission_probability(
                retrieval.item_id, deterministic_fine, draw, rate,
            ),
            targeted_position_probability(
                exposed_item, deterministic_exposed, draw, rate,
            ),
            targeted_admission_probability(
                retrieval.item_id, deterministic_exposed, draw, rate,
            ),
            selection_kind,
            torch.full_like(requests.request_id, rate, dtype=torch.float),
            targeted_slate_log_probability(
                exposed_item, deterministic_exposed, draw, rate,
            ),
        )

    @staticmethod
    def _selection_metadata(
        requests: PlatformRequestBatch,
        retrieval: RetrievalResult,
        deterministic_coarse: torch.Tensor,
        deterministic_fine: torch.Tensor,
        deterministic_exposed: torch.Tensor,
        exposed_item: torch.Tensor,
        randomized: torch.Tensor,
        policy: CascadePolicy,
    ) -> tuple[torch.Tensor, ...]:
        eligible_count = (retrieval.item_id >= 0).sum(dim=1)
        coarse_probability = mixture_admission_probability(
            retrieval.item_id, deterministic_coarse, policy.exploration_rate,
        )
        fine_probability = mixture_admission_probability(
            retrieval.item_id, deterministic_fine, policy.exploration_rate,
        )
        exposure_probability = mixture_position_probability(
            exposed_item,
            deterministic_exposed,
            eligible_count,
            policy.exploration_rate,
        )
        candidate_exposure_probability = mixture_admission_probability(
            retrieval.item_id,
            deterministic_exposed,
            policy.exploration_rate,
        )
        slate_probability = mixture_slate_log_probability(
            exposed_item,
            deterministic_exposed,
            eligible_count,
            policy.exploration_rate,
        )
        selection_kind = torch.where(
            randomized,
            torch.full_like(
                requests.request_id, int(SelectionPolicyKind.RANDOMIZED),
            ),
            torch.full_like(
                requests.request_id, int(SelectionPolicyKind.DETERMINISTIC),
            ),
        )
        rate = torch.full_like(
            requests.request_id, policy.exploration_rate, dtype=torch.float,
        )
        return (
            coarse_probability,
            fine_probability,
            exposure_probability,
            candidate_exposure_probability,
            selection_kind,
            rate,
            slate_probability,
        )

    def _fine_scores(
        self,
        features: FeatureTensorBatch,
        item_id: torch.Tensor,
        requests: PlatformRequestBatch,
        policy: CascadePolicy,
    ) -> torch.Tensor:
        if policy.fine_version_id == 0:
            priority = -torch.arange(
                item_id.shape[1], device=item_id.device, dtype=torch.float,
            )[None].expand_as(item_id)
            return priority.masked_fill(item_id < 0, -torch.inf)
        scorer = self._fine_scorers.get(policy.fine_version_id)
        if scorer is not None:
            score = scorer.score(features, requests.surface)
            if score.shape != item_id.shape:
                raise ValueError("learned fine scorer returned an invalid shape")
            return score.to(item_id.device).masked_fill(item_id < 0, -torch.inf)
        dense = features.dense[:, :, :10]
        weight = torch.tensor(policy.fine_weights, device=item_id.device)
        score = torch.einsum("bkd,d->bk", dense, weight)
        score += policy.cross_weight * torch.sin(1.7 * dense[:, :, 0]) * dense[:, :, 1]
        score += policy.sequence_weight * features.dense[:, :, 10]
        return score.masked_fill(item_id < 0, -torch.inf)

    def _coarse_scores(
        self,
        features: FeatureTensorBatch,
        item_id: torch.Tensor,
        requests: PlatformRequestBatch,
        policy: CascadePolicy,
    ) -> torch.Tensor:
        if policy.coarse_version_id == 0:
            priority = -torch.arange(
                item_id.shape[1], device=item_id.device, dtype=torch.float,
            )[None].expand_as(item_id)
            return priority.masked_fill(item_id < 0, -torch.inf)
        scorer = self._coarse_scorers.get(policy.coarse_version_id)
        if scorer is not None:
            score = scorer.score(features, requests.surface)
            if score.shape != item_id.shape:
                raise ValueError("learned coarse scorer returned an invalid shape")
            return score.to(item_id.device).masked_fill(item_id < 0, -torch.inf)
        dense = features.dense[:, :, :10]
        weight = torch.tensor(policy.coarse_weights, device=item_id.device)
        score = torch.einsum("bkd,d->bk", dense, weight)
        score += policy.cross_weight * (
            dense[:, :, 0] * dense[:, :, 1]
            + dense[:, :, 4] * dense[:, :, 5]
        )
        return score.masked_fill(item_id < 0, -torch.inf)

    @staticmethod
    def _prefix(item_id: torch.Tensor, limit: int) -> torch.Tensor:
        width = min(limit, item_id.shape[1])
        selected = item_id[:, :width]
        if width < limit:
            selected = torch.nn.functional.pad(
                selected, (0, limit - width), value=-1,
            )
        return selected

    @staticmethod
    def _select_features(
        child: torch.Tensor,
        parent: torch.Tensor,
        features: FeatureTensorBatch,
    ) -> FeatureTensorBatch:
        match = child[:, :, None] == parent[:, None, :]
        location = match.float().argmax(dim=2)
        valid = match.any(dim=2)

        def gather(value: torch.Tensor) -> torch.Tensor:
            index = location[:, :, None].expand(-1, -1, value.shape[2])
            selected = torch.gather(value, 1, index)
            return selected.masked_fill(~valid[:, :, None], 0)

        return FeatureTensorBatch(
            dense=gather(features.dense),
            sparse_fids=gather(features.sparse_fids),
            sparse_buckets=gather(features.sparse_buckets),
            manifest_hash=features.manifest_hash,
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
            prior_ad = (
                (self.catalog.content_kind[prior_item] == int(ContentKind.AD))
                & prior_valid
            ).sum(dim=1)
            ad_blocked = (
                (kind == int(ContentKind.AD))
                & (prior_ad >= 1)[:, None]
            )
            adjusted = (
                score
                - self.config.creator_penalty * creator_count.sum(dim=2)
                - self.config.kind_penalty * kind_count.sum(dim=2)
            ).masked_fill(~available | ad_blocked, -torch.inf)
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
