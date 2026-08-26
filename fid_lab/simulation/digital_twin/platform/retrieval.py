"""Independent observable retrieval routes with FAISS and sparse co-visits."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ...randomness.counter import uniform
from .retrieval_merge import reciprocal_rank_fusion
from .routes.exposure import exposed_in_current_session, recently_exposed
from .routes.popular import interest_popular_candidates, popular_candidates
from .routes.posting import posting_route_scores
from .routes.commerce import inventory_eligible
from .indexes.ann import FaissItemIndex
from .indexes.contracts import LearnedRetriever, RetrievalConfig
from .indexes.graph import CoVisitGraphIndex, STRONG_GRAPH_EVENTS

from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    ContentKind,
    EventType,
    PlatformRequestBatch,
    Surface,
)
from .projection import ITEM_COUNTER_EVENTS, PlatformProjectionState
from .lifecycle import ContentLifecycle
from .routes import (
    BUSINESS_ROUTE_NAMES,
    MAIN_FEED_LIFECYCLES,
    ROUTE_NAMES,
    build_feed_route_signals,
    surface_eligibility,
)


@dataclass(frozen=True)
class RetrievalResult:
    item_id: torch.Tensor
    route_bits: torch.Tensor
    score: torch.Tensor
    sampling_probability: torch.Tensor
    route_item_id: torch.Tensor
    route_score: torch.Tensor
    route_valid: torch.Tensor
    index_version: str


class MultiRouteRetriever:
    def __init__(self, catalog: PublicCatalog, config: RetrievalConfig):
        self.catalog = catalog
        self.config = config
        self.device = catalog.item_id.device
        self.faiss = FaissItemIndex(catalog, config)
        self.graph = CoVisitGraphIndex(
            len(catalog.item_id), config.graph_neighbors, self.device,
        )
        self._last_refresh = -1
        self._global_query = torch.nn.functional.normalize(
            catalog.content_embedding.mean(dim=0, keepdim=True), dim=1,
        )
        topics = int(catalog.topic_id.max()) + 1
        topic_embedding = torch.zeros(
            topics,
            catalog.content_embedding.shape[1],
            device=self.device,
        )
        topic_count = torch.zeros(topics, device=self.device)
        topic_embedding.index_add_(
            0, catalog.topic_id, catalog.content_embedding,
        )
        topic_count.index_add_(
            0, catalog.topic_id, torch.ones_like(catalog.topic_id).float(),
        )
        self._topic_query = torch.nn.functional.normalize(
            topic_embedding / topic_count.clamp_min(1.0)[:, None], dim=1,
        )
        self.learned_retriever: LearnedRetriever | None = None

    @property
    def index_version(self) -> str:
        if self.learned_retriever is not None:
            return self.learned_retriever.index_version
        return self.faiss.version

    def install_learned_retriever(self, retriever: LearnedRetriever | None) -> None:
        self.learned_retriever = retriever

    @property
    def route_names(self) -> tuple[str, ...]:
        return ROUTE_NAMES

    def ingest(self, events: AppEventBatch) -> None:
        self.graph.update(events)

    def refresh(
        self, state: PlatformProjectionState, logical_time: int,
    ) -> None:
        if (
            self._last_refresh >= 0
            and logical_time - self._last_refresh < self.config.refresh_interval
        ):
            return
        version = f"observable-index-t{logical_time}"
        self.faiss.sync(state.item_active, version)
        self.graph.refresh(version)
        self._last_refresh = logical_time

    def query_embedding(
        self, requests: PlatformRequestBatch, state: PlatformProjectionState,
    ) -> torch.Tensor:
        history = state.user_history_item[requests.user_id]
        history_event = state.user_history_event_type[requests.user_id]
        positive_events = torch.tensor(
            [int(event_type) for event_type in STRONG_GRAPH_EVENTS],
            device=self.device,
        )
        valid = (history >= 0) & torch.isin(history_event, positive_events)
        embedding = self.catalog.content_embedding[history.clamp_min(0)]
        history_time = state.user_history_event_time[requests.user_id]
        age = (requests.event_time[:, None] - history_time).clamp_min(0)
        decay = torch.exp2(
            -age.float() / float(self.config.interest_half_life_ticks)
        )
        weight = valid.float() * decay
        summed = (embedding * weight[:, :, None]).sum(dim=1)
        count = weight.sum(dim=1, keepdim=True)
        query = summed / count.clamp_min(1e-6)
        query = torch.where(
            (count > 0).expand_as(query), query, self._global_query,
        )
        return torch.nn.functional.normalize(query, dim=1)

    def _last_item(
        self, requests: PlatformRequestBatch, state: PlatformProjectionState,
    ) -> torch.Tensor:
        item = state.user_history_item[requests.user_id]
        event_type = state.user_history_event_type[requests.user_id]
        event_time = state.user_history_event_time[requests.user_id]
        strong_events = torch.tensor(
            [int(value) for value in STRONG_GRAPH_EVENTS], device=self.device,
        )
        valid = (item >= 0) & torch.isin(event_type, strong_events)
        latest = event_time.masked_fill(~valid, -1).argmax(dim=1)
        selected = torch.gather(item, 1, latest[:, None]).squeeze(1)
        return torch.where(
            valid.any(dim=1), selected, torch.full_like(selected, -1),
        )

    @staticmethod
    def _last_followed_creator(
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
    ) -> torch.Tensor:
        cursor = state.user_followed_creator_cursor[requests.user_id]
        slot = torch.remainder(
            cursor - 1, state.user_followed_creator.shape[1],
        )
        creator = state.user_followed_creator[requests.user_id, slot]
        return torch.where(
            cursor > 0, creator, torch.full_like(creator, -1),
        )

    def _top_by_group(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        score: torch.Tensor,
        group: torch.Tensor,
        item_group: torch.Tensor,
        *,
        require_query: bool = False,
        allowed_lifecycle: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, limit = len(requests.user_id), self.config.route_k
        item = torch.full((rows, limit), -1, device=self.device, dtype=torch.long)
        values = torch.full((rows, limit), -torch.inf, device=self.device)
        for key in torch.unique(group).tolist():
            selected_rows = group == key
            if key < 0 or (require_query and not selected_rows.any()):
                continue
            eligible = (
                state.item_active
                & (item_group == key)
            )
            if allowed_lifecycle is not None:
                eligible &= self._lifecycle_mask(
                    state, allowed_lifecycle,
                )
            row_index = torch.where(selected_rows)[0]
            for surface in torch.unique(requests.surface[row_index]).tolist():
                target = row_index[requests.surface[row_index] == surface]
                mask = eligible & surface_eligibility(
                    surface, self.catalog.content_kind,
                )
                count = min(limit, int(mask.sum()))
                if not count:
                    continue
                candidate = torch.where(mask)[0]
                top = torch.topk(score[candidate], count).indices
                chosen = candidate[top]
                item[target, :count] = chosen[None]
                values[target, :count] = score[chosen][None]
        return item, values

    @staticmethod
    def _lifecycle_mask(
        state: PlatformProjectionState,
        allowed: tuple[int, ...],
    ) -> torch.Tensor:
        result = torch.zeros_like(state.item_active)
        for lifecycle in allowed:
            result |= state.item_lifecycle == lifecycle
        return result

    def _top_for_surface(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        score: torch.Tensor,
        surface: Surface,
        *,
        allowed_lifecycle: tuple[int, ...] | None = None,
        extra_eligible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = len(requests.user_id)
        item = torch.full(
            (rows, self.config.route_k),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        values = torch.full_like(item, -torch.inf, dtype=torch.float)
        selected = requests.surface == int(surface)
        eligible = state.item_active & surface_eligibility(
            int(surface), self.catalog.content_kind,
        )
        if allowed_lifecycle is not None:
            eligible &= self._lifecycle_mask(state, allowed_lifecycle)
        if extra_eligible is not None:
            eligible &= extra_eligible
        count = min(self.config.route_k, int(eligible.sum()))
        if selected.any() and count:
            candidates = torch.where(eligible)[0]
            top = torch.topk(score[candidates], count).indices
            chosen = candidates[top]
            item[selected, :count] = chosen[None]
            values[selected, :count] = score[chosen][None]
        return item, values

    def _uniform_lifecycle_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        score: torch.Tensor,
        lifecycle: ContentLifecycle | tuple[ContentLifecycle, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = len(requests.user_id)
        item = torch.full(
            (rows, self.config.route_k),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        values = torch.full_like(item, -torch.inf, dtype=torch.float)
        feed = requests.surface == int(Surface.FEED)
        lifecycles = (
            lifecycle if isinstance(lifecycle, tuple) else (lifecycle,)
        )
        eligible = (
            state.item_active
            & self._lifecycle_mask(
                state, tuple(int(value) for value in lifecycles),
            )
            & surface_eligibility(
                int(Surface.FEED), self.catalog.content_kind,
            )
        )
        candidates = torch.where(eligible)[0]
        width = min(self.config.route_k, len(candidates))
        if not feed.any() or not width:
            return item, values
        logical_time = int(requests.event_time.min())
        random_key = uniform(
            candidates,
            logical_time,
            313,
            self.config.selection_seed,
        )
        candidates = candidates[torch.argsort(random_key, stable=True)]
        start = torch.remainder(
            requests.request_id[feed] * 503 + requests.user_id[feed] * 1_009,
            len(candidates),
        )
        offset = torch.arange(width, device=self.device)[None]
        location = torch.remainder(start[:, None] + offset, len(candidates))
        chosen = candidates[location]
        item[feed, :width] = chosen
        values[feed, :width] = score[chosen]
        return item, values

    def retrieve(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        enabled_routes: tuple[str, ...] = ROUTE_NAMES,
        route_weights: tuple[float, ...] = (),
        *,
        feed_exposure_dedup_ticks: int = 0,
        feed_session_dedup: bool = False,
        commerce_require_inventory: bool = False,
        commerce_min_inventory: float = 0.0,
    ) -> RetrievalResult:
        unknown = set(enabled_routes) - set(ROUTE_NAMES)
        if unknown:
            raise ValueError(f"unknown retrieval routes: {sorted(unknown)}")
        if not enabled_routes:
            raise ValueError("at least one retrieval route must be enabled")
        if route_weights and len(route_weights) != len(ROUTE_NAMES):
            raise ValueError("route weights must align with the route registry")
        indexed_routes = {"recent_ann", "recent_graph", "search_semantic", "retarget"}
        if set(enabled_routes) & (indexed_routes | set(BUSINESS_ROUTE_NAMES)):
            self.refresh(state, int(requests.event_time.min()))
        route_item, route_score = self._route_candidates(
            requests,
            state,
            enabled_routes,
            commerce_require_inventory=commerce_require_inventory,
            commerce_min_inventory=commerce_min_inventory,
        )
        route_valid = route_item >= 0
        safe_route_item = route_item.clamp_min(0)
        ad_candidate = self.catalog.content_kind[safe_route_item] == int(
            ContentKind.AD
        )
        ads_route = torch.zeros(
            len(ROUTE_NAMES), device=self.device, dtype=torch.bool,
        )
        ads_route[ROUTE_NAMES.index("ads_auction")] = True
        route_valid &= ~(ad_candidate & ~ads_route[None, :, None])
        route_valid &= ~(~ad_candidate & ads_route[None, :, None])
        if feed_exposure_dedup_ticks:
            repeated = recently_exposed(
                requests,
                state,
                route_item,
                feed_exposure_dedup_ticks,
            )
            route_valid &= ~repeated
        if feed_session_dedup:
            repeated = exposed_in_current_session(
                requests,
                state,
                route_item,
            )
            route_valid &= ~repeated
        inventory_floor = max(
            commerce_min_inventory,
            0.0 if not commerce_require_inventory else 1e-12,
        )
        if commerce_require_inventory or commerce_min_inventory > 0.0:
            route_valid &= inventory_eligible(
                requests, self.catalog, state, route_item, inventory_floor,
            )
        enabled = torch.tensor(
            [name in enabled_routes for name in ROUTE_NAMES],
            device=self.device,
            dtype=torch.bool,
        )
        route_valid &= enabled[None, :, None]
        route_item = torch.where(
            route_valid, route_item, torch.full_like(route_item, -1),
        )
        route_score = torch.where(
            route_valid,
            route_score,
            torch.full_like(route_score, -torch.inf),
        )
        if len(enabled_routes) == 1:
            route_index = ROUTE_NAMES.index(enabled_routes[0])
            merged_item = route_item[:, route_index]
            merged_score = route_score[:, route_index]
            route_bits = torch.where(
                route_valid[:, route_index],
                torch.full_like(merged_item, 1 << route_index),
                torch.zeros_like(merged_item),
            )
        else:
            merge_weight = torch.tensor(
                route_weights or (1.0,) * len(ROUTE_NAMES),
                device=self.device,
            )
            merged_item, merged_score, route_bits = reciprocal_rank_fusion(
                route_item,
                route_valid & (merge_weight > 0.0)[None, :, None],
                merge_weight,
                reciprocal_rank_constant=self.config.reciprocal_rank_constant,
                merged_k=self.config.merged_k,
            )
        return RetrievalResult(
            item_id=merged_item,
            route_bits=route_bits,
            score=merged_score,
            sampling_probability=torch.where(
                merged_item >= 0,
                torch.ones_like(merged_score),
                torch.zeros_like(merged_score),
            ),
            route_item_id=route_item,
            route_score=route_score,
            route_valid=route_valid,
            index_version=self.index_version,
        )

    def _route_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        enabled_routes: tuple[str, ...],
        *,
        commerce_require_inventory: bool = False,
        commerce_min_inventory: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enabled = set(enabled_routes)
        routes = self._feed_route_candidates(requests, state, enabled)
        business_enabled = enabled - set(routes)
        if business_enabled:
            business = self._business_route_candidates(
                requests,
                state,
                commerce_require_inventory=commerce_require_inventory,
                commerce_min_inventory=commerce_min_inventory,
            )
            routes.update({
                name: value for name, value in business.items()
                if name in business_enabled
            })
        missing = enabled - set(routes)
        extra = set(routes) - set(ROUTE_NAMES)
        if missing or extra:
            raise ValueError(
                f"route registry mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        empty_item = torch.full(
            (len(requests.user_id), self.config.route_k),
            -1, device=self.device, dtype=torch.long,
        )
        empty_score = torch.full_like(empty_item, -torch.inf, dtype=torch.float)
        empty = (empty_item, empty_score)
        return (
            torch.stack(tuple(
                routes.get(name, empty)[0] for name in ROUTE_NAMES
            ), dim=1),
            torch.stack(tuple(
                routes.get(name, empty)[1] for name in ROUTE_NAMES
            ), dim=1),
        )

    def _feed_route_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        enabled: set[str],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        feed_enabled = enabled & {
            "random", "popular", "interest_popular", "blended_popular",
            "recent_ann", "recent_graph", "following", "cold_start", "hot",
            "evergreen",
        }
        if not feed_enabled:
            return {}
        routes = {}
        if "recent_ann" in feed_enabled:
            routes["recent_ann"] = self._recent_ann_candidates(requests, state)
        if "recent_graph" in feed_enabled:
            routes["recent_graph"] = self._recent_graph_candidates(requests, state)
        signal_routes = feed_enabled - {"recent_ann", "recent_graph"}
        if signal_routes:
            signals = build_feed_route_signals(
                self.catalog, state, requests.event_time.max(),
            )
            routes.update(self._signal_route_candidates(
                requests, state, signals, signal_routes,
            ))
        return routes

    def _recent_ann_candidates(self, requests, state):
        if self.learned_retriever is None:
            query = self.query_embedding(requests, state)
            item, score = self.faiss.search(
                query, self.config.route_k * self.config.ann_oversample,
            )
        else:
            item, score = self.learned_retriever.retrieve(
                requests, state, self.config.route_k * self.config.ann_oversample,
            )
        safe = item.clamp_min(0)
        user_country = state.user_country[requests.user_id]
        country_match = (
            (user_country < 0)[:, None]
            | (self.catalog.country[safe] == user_country[:, None])
        )
        item = torch.where(country_match, item, torch.full_like(item, -1))
        score = score.masked_fill(item < 0, -torch.inf)
        return self._filter_and_trim(
            requests,
            state,
            item,
            score,
            required_surface=Surface.FEED,
            allowed_lifecycle=(int(ContentLifecycle.RECENT),),
        )

    def _recent_graph_candidates(self, requests, state):
        last_item = self._last_item(requests, state)
        item = self.graph.neighbor[last_item.clamp_min(0)]
        score = self.graph.score[last_item.clamp_min(0)]
        safe = item.clamp_min(0)
        user_country = state.user_country[requests.user_id]
        country_match = (
            (user_country < 0)[:, None]
            | (self.catalog.country[safe] == user_country[:, None])
        )
        item = torch.where(
            (last_item >= 0)[:, None] & country_match,
            item,
            torch.full_like(item, -1),
        )
        score = score.masked_fill(item < 0, -torch.inf)
        return self._filter_and_trim(
            requests,
            state,
            item,
            score,
            required_surface=Surface.FEED,
            allowed_lifecycle=(int(ContentLifecycle.RECENT),),
        )

    def _signal_route_candidates(
        self, requests, state, signals, enabled,
    ):
        routes = {}
        if "random" in enabled:
            routes["random"] = self._uniform_lifecycle_candidates(
                requests, state, signals.random, MAIN_FEED_LIFECYCLES,
            )
        if "popular" in enabled:
            routes["popular"] = popular_candidates(
                self.catalog, self.config, requests, state, signals.popular,
            )
        if "interest_popular" in enabled:
            routes["interest_popular"] = interest_popular_candidates(
                self.catalog, self.config, requests, state, signals.popular,
            )
        if "blended_popular" in enabled:
            routes["blended_popular"] = interest_popular_candidates(
                self.catalog,
                self.config,
                requests,
                state,
                signals.popular,
                interest_fraction=self.config.popular_interest_fraction,
            )
        if "cold_start" in enabled:
            routes["cold_start"] = self._uniform_lifecycle_candidates(
                requests, state, signals.cold_start, ContentLifecycle.COLD_START,
            )
        if "hot" in enabled:
            routes["hot"] = self._top_for_surface(
                requests, state, signals.hot, Surface.FEED,
                allowed_lifecycle=(int(ContentLifecycle.HOT),),
            )
        if "evergreen" in enabled:
            routes["evergreen"] = self._top_for_surface(
                requests, state, signals.evergreen, Surface.FEED,
                allowed_lifecycle=(int(ContentLifecycle.EVERGREEN),),
            )
        if "following" in enabled:
            routes["following"] = self._following_candidates(
                requests, state, signals.following,
            )
        return routes

    def _following_candidates(self, requests, state, score):
        creator = self._last_followed_creator(requests, state)
        creator = torch.where(
            requests.surface == int(Surface.FEED),
            creator,
            torch.full_like(creator, -1),
        )
        return self._top_by_group(
            requests,
            state,
            score,
            creator,
            state.item_creator_id,
            allowed_lifecycle=(
                int(ContentLifecycle.COLD_START),
                int(ContentLifecycle.RECENT),
                int(ContentLifecycle.HOT),
                int(ContentLifecycle.EVERGREEN),
            ),
        )

    def _business_route_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        *,
        commerce_require_inventory: bool = False,
        commerce_min_inventory: float = 0.0,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        last_item = self._last_item(requests, state)
        impression = state.item_event_counts[
            :, ITEM_COUNTER_EVENTS.index(EventType.IMPRESSION)
        ]
        popular_score = torch.log1p(impression)
        engagement_rate = state.item_recent_engagements / (
            state.item_recent_impressions.clamp_min(1.0)
        )
        region = state.user_region[requests.user_id]
        local_region = torch.where(
            requests.surface == int(Surface.LOCAL),
            region,
            torch.full_like(region, -1),
        )
        local_item, local_score = self._top_by_group(
            requests,
            state,
            self.catalog.quality_prior + 0.2 * state.item_inventory,
            local_region,
            state.item_region,
        )
        posting, posting_diverse = self._posting_candidates(
            requests, state, impression, region,
        )
        commerce_eligible = self.catalog.content_kind != int(ContentKind.AD)
        if commerce_require_inventory or commerce_min_inventory > 0.0:
            commerce_eligible &= (
                (self.catalog.content_kind != int(ContentKind.PRODUCT))
                | (
                    state.item_inventory
                    > max(
                        commerce_min_inventory,
                        0.0 if not commerce_require_inventory else 1e-12,
                    )
                )
            )
        commerce_item, commerce_score = self._top_for_surface(
            requests,
            state,
            0.45 * self.catalog.quality_prior
            + 0.35 * state.item_inventory
            + 0.20 * torch.log1p(state.item_bid),
            Surface.COMMERCE,
            extra_eligible=commerce_eligible,
        )
        live_item, live_score = self._top_for_surface(
            requests,
            state,
            0.55 * self.catalog.quality_prior
            + 0.45 * engagement_rate,
            Surface.LIVE,
        )
        ads_item, ads_score = self._ads_route_candidates(requests, state)
        search_item, search_score = self._top_by_group(
            requests,
            state,
            self.catalog.quality_prior + 0.15 * popular_score,
            requests.query_topic,
            self.catalog.topic_id,
            require_query=True,
        )
        semantic_query = self._topic_query[requests.query_topic.clamp_min(0)]
        semantic_item, semantic_score = self.faiss.search(
            semantic_query, self.config.route_k * self.config.ann_oversample,
        )
        semantic_item, semantic_score = self._filter_and_trim(
            requests,
            state,
            semantic_item,
            semantic_score,
            required_surface=Surface.SEARCH,
        )
        no_query = requests.query_topic < 0
        semantic_item[no_query] = -1
        semantic_score[no_query] = -torch.inf
        retarget_query = self.catalog.content_embedding[last_item.clamp_min(0)]
        retarget_item, retarget_score = self.faiss.search(
            retarget_query, self.config.route_k * self.config.ann_oversample,
        )
        retarget_item = torch.where(
            (last_item >= 0)[:, None],
            retarget_item,
            torch.full_like(retarget_item, -1),
        )
        retarget_item, retarget_score = self._filter_and_trim(
            requests, state, retarget_item, retarget_score,
        )
        return {
            "local_geo": (local_item, local_score),
            "posting_context": posting,
            "posting_diverse": posting_diverse,
            "commerce_intent": (commerce_item, commerce_score),
            "live_now": (live_item, live_score),
            "ads_auction": (ads_item, ads_score),
            "search": (search_item, search_score),
            "search_semantic": (semantic_item, semantic_score),
            "retarget": (retarget_item, retarget_score),
        }

    def _posting_candidates(self, requests, state, impression, region):
        posting_region = torch.where(
            requests.surface == int(Surface.POSTING),
            region,
            torch.full_like(region, -1),
        )
        baseline, diverse = posting_route_scores(
            self.catalog, state, impression,
        )
        return (
            self._top_by_group(
                requests, state, baseline, posting_region, state.item_region,
            ),
            self._top_by_group(
                requests, state, diverse, posting_region, state.item_region,
            ),
        )

    def _ads_route_candidates(self, requests, state):
        ad = self.catalog.content_kind == int(ContentKind.AD)
        advertiser = self.catalog.advertiser_id
        eligible = (
            ad
            & (state.advertiser_bid[advertiser] > 0.0)
            & (
                state.advertiser_budget[advertiser]
                >= state.advertiser_bid[advertiser]
            )
        )
        return self._top_for_surface(
            requests,
            state,
            0.55 * self.catalog.quality_prior
            + 0.45 * torch.log1p(state.item_bid),
            Surface.FEED,
            extra_eligible=eligible,
        )

    def _filter_and_trim(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item: torch.Tensor,
        score: torch.Tensor,
        *,
        required_surface: Surface | None = None,
        allowed_lifecycle: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = item.clamp_min(0)
        valid = item >= 0
        valid &= state.item_active[safe]
        valid &= surface_eligibility(
            requests.surface,
            self.catalog.content_kind[safe],
        )
        if required_surface is not None:
            valid &= (
                requests.surface == int(required_surface)
            )[:, None]
        if allowed_lifecycle is not None:
            valid &= self._lifecycle_mask(
                state, allowed_lifecycle,
            )[safe]
        history = state.user_history_item[requests.user_id]
        valid &= ~(safe[:, :, None] == history[:, None, :]).any(dim=2)
        score = score.masked_fill(~valid, -torch.inf)
        width = min(self.config.route_k, item.shape[1])
        order = torch.topk(score, width, dim=1).indices
        chosen_item = torch.gather(item, 1, order)
        chosen_score = torch.gather(score, 1, order)
        chosen_item = torch.where(
            torch.isfinite(chosen_score), chosen_item, torch.full_like(chosen_item, -1)
        )
        if width < self.config.route_k:
            padding = self.config.route_k - width
            chosen_item = torch.nn.functional.pad(
                chosen_item, (0, padding), value=-1,
            )
            chosen_score = torch.nn.functional.pad(
                chosen_score, (0, padding), value=-torch.inf,
            )
        return chosen_item, chosen_score
