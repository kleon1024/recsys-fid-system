"""Independent observable retrieval routes with FAISS and sparse co-visits."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
from typing import Protocol

import numpy as np
import scipy.sparse
import torch

from .retrieval_merge import reciprocal_rank_fusion
from .routes.exposure import exposed_in_current_session, recently_exposed

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
    MAIN_FEED_LIFECYCLES,
    ROUTE_NAMES,
    build_feed_route_signals,
    surface_eligibility,
)


@dataclass(frozen=True)
class RetrievalConfig:
    route_k: int = 32
    merged_k: int = 128
    ann_oversample: int = 4
    graph_neighbors: int = 32
    reciprocal_rank_constant: float = 20.0
    refresh_interval: int = 8
    hnsw_neighbors: int = 24
    hnsw_ef_search: int = 64

    def __post_init__(self):
        dimensions = (
            self.route_k,
            self.merged_k,
            self.ann_oversample,
            self.graph_neighbors,
            self.refresh_interval,
            self.hnsw_neighbors,
            self.hnsw_ef_search,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("retrieval dimensions must be positive")


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


class LearnedRetriever(Protocol):
    serving_version_id: int

    @property
    def index_version(self) -> str: ...

    def retrieve(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class FaissItemIndex:
    def __init__(self, catalog: PublicCatalog, config: RetrievalConfig):
        self.catalog = catalog
        self.config = config
        self.version = "unbuilt"
        self.backend = os.environ.get(
            "FID_ANN_BACKEND", "torch" if sys.platform == "darwin" else "faiss"
        )
        if self.backend not in {"faiss", "torch"}:
            raise ValueError("FID_ANN_BACKEND must be faiss or torch")
        self._index: object | None = None
        self._torch_item: torch.Tensor | None = None
        self._torch_embedding: torch.Tensor | None = None
        self._indexed_active = torch.zeros_like(catalog.active)

    def sync(self, active: torch.Tensor, version: str) -> None:
        if self.backend == "torch":
            self._torch_item = self.catalog.item_id[active]
            self._torch_embedding = self.catalog.content_embedding[active]
            self._indexed_active = active.clone()
        else:
            faiss = importlib.import_module("faiss")
            faiss.omp_set_num_threads(int(os.environ.get("FID_FAISS_THREADS", "1")))
            if (self._indexed_active & ~active).any():
                self._index = None
                self._indexed_active.zero_()
            new_item = active & ~self._indexed_active
            item = self.catalog.item_id[new_item].detach().cpu().numpy().astype("int64")
            vectors = self.catalog.content_embedding[new_item].detach().cpu().numpy()
            vectors = np.ascontiguousarray(vectors.astype("float32"))
            if self._index is None:
                base = faiss.IndexHNSWFlat(
                    self.catalog.content_embedding.shape[1],
                    self.config.hnsw_neighbors,
                    faiss.METRIC_INNER_PRODUCT,
                )
                base.hnsw.efConstruction = max(
                    2 * self.config.hnsw_neighbors, 40,
                )
                base.hnsw.efSearch = self.config.hnsw_ef_search
                self._index = faiss.IndexIDMap2(base)
            if len(item):
                self._index.add_with_ids(vectors, item)
                self._indexed_active |= new_item
        self.version = version

    def search(
        self, query: torch.Tensor, limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "torch":
            if self._torch_embedding is None or self._torch_item is None:
                raise ValueError("Torch item index has not been built")
            count = min(limit, len(self._torch_item))
            score, location = torch.topk(
                query @ self._torch_embedding.T, count, dim=1,
            )
            return self._torch_item[location], score
        if self._index is None:
            raise ValueError("FAISS item index has not been built")
        query_np = np.ascontiguousarray(
            query.detach().cpu().numpy().astype("float32")
        )
        score, item = self._index.search(query_np, limit)
        return (
            torch.from_numpy(item).to(query.device),
            torch.from_numpy(score).to(query.device),
        )


class CoVisitGraphIndex:
    """Sparse behavioral graph rebuilt off the request path with SciPy CSR."""

    def __init__(self, items: int, neighbors: int, device: torch.device):
        self.items = items
        self.neighbor_count = neighbors
        self.device = device
        self._matrix = scipy.sparse.csr_matrix((items, items), dtype=np.float32)
        self._pending_source: list[np.ndarray] = []
        self._pending_target: list[np.ndarray] = []
        self.neighbor = torch.full(
            (items, neighbors), -1, device=device, dtype=torch.long,
        )
        self.score = torch.zeros(items, neighbors, device=device)
        self.version = "graph-empty"

    def update(self, events: AppEventBatch) -> None:
        dwell = events.event(EventType.DWELL) & (events.item_id >= 0)
        if int(dwell.sum()) < 2:
            return
        request = events.request_id[dwell]
        position = events.position[dwell]
        item = events.item_id[dwell]
        order = torch.argsort(position, stable=True)
        order = order[torch.argsort(request[order], stable=True)]
        request, item = request[order], item[order]
        adjacent = request[1:] == request[:-1]
        source, target = item[:-1][adjacent], item[1:][adjacent]
        valid = source != target
        source, target = source[valid], target[valid]
        if not len(source):
            return
        self._pending_source.append(torch.cat((source, target)).cpu().numpy())
        self._pending_target.append(torch.cat((target, source)).cpu().numpy())

    def refresh(self, version: str) -> None:
        if not self._pending_source:
            return
        source = np.concatenate(self._pending_source)
        target = np.concatenate(self._pending_target)
        increment = scipy.sparse.coo_matrix(
            (np.ones(len(source), dtype=np.float32), (source, target)),
            shape=(self.items, self.items),
        ).tocsr()
        self._matrix = self._matrix + increment
        self._pending_source.clear()
        self._pending_target.clear()
        rows = np.unique(source)
        for row in rows:
            start, end = self._matrix.indptr[row:row + 2]
            columns = self._matrix.indices[start:end]
            values = self._matrix.data[start:end]
            if not len(columns):
                continue
            keep = np.argsort(-values, kind="stable")[: self.neighbor_count]
            width = len(keep)
            self.neighbor[row].fill_(-1)
            self.score[row].zero_()
            self.neighbor[row, :width] = torch.from_numpy(
                columns[keep].astype("int64")
            ).to(self.device)
            self.score[row, :width] = torch.from_numpy(
                values[keep]
            ).to(self.device)
        self.version = version


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
        valid = history >= 0
        embedding = self.catalog.content_embedding[history.clamp_min(0)]
        summed = (embedding * valid[:, :, None]).sum(dim=1)
        count = valid.sum(dim=1, keepdim=True)
        query = summed / count.clamp_min(1)
        query = torch.where(
            (count > 0).expand_as(query), query, self._global_query,
        )
        return torch.nn.functional.normalize(query, dim=1)

    def _last_item(
        self, requests: PlatformRequestBatch, state: PlatformProjectionState,
    ) -> torch.Tensor:
        cursor = state.user_history_cursor[requests.user_id]
        slot = torch.remainder(
            cursor - 1, state.user_history_item.shape[1]
        )
        item = state.user_history_item[requests.user_id, slot]
        return torch.where(cursor > 0, item, torch.full_like(item, -1))

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
        count = min(self.config.route_k, int(eligible.sum()))
        if selected.any() and count:
            candidates = torch.where(eligible)[0]
            top = torch.topk(score[candidates], count).indices
            chosen = candidates[top]
            item[selected, :count] = chosen[None]
            values[selected, :count] = score[chosen][None]
        return item, values

    def _rotating_lifecycle_candidates(
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
        *,
        feed_exposure_dedup_ticks: int = 0,
        feed_session_dedup: bool = False,
    ) -> RetrievalResult:
        unknown = set(enabled_routes) - set(ROUTE_NAMES)
        if unknown:
            raise ValueError(f"unknown retrieval routes: {sorted(unknown)}")
        if not enabled_routes:
            raise ValueError("at least one retrieval route must be enabled")
        if self.faiss.version == "unbuilt":
            self.refresh(state, int(requests.event_time.min()))
        route_item, route_score = self._route_candidates(requests, state)
        route_valid = route_item >= 0
        if feed_exposure_dedup_ticks:
            repeated = recently_exposed(
                requests,
                state,
                route_item,
                feed_exposure_dedup_ticks,
            )
            video = self.catalog.content_kind[route_item.clamp_min(0)] == int(
                ContentKind.SHORT_VIDEO
            )
            route_valid &= ~(repeated & video)
        if feed_session_dedup:
            repeated = exposed_in_current_session(
                requests,
                state,
                route_item,
            )
            video = self.catalog.content_kind[route_item.clamp_min(0)] == int(
                ContentKind.SHORT_VIDEO
            )
            route_valid &= ~(repeated & video)
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
        merged_item, merged_score, route_bits = reciprocal_rank_fusion(
            route_item,
            route_valid,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routes = {
            **self._feed_route_candidates(requests, state),
            **self._business_route_candidates(requests, state),
        }
        missing = set(ROUTE_NAMES) - set(routes)
        extra = set(routes) - set(ROUTE_NAMES)
        if missing or extra:
            raise ValueError(
                f"route registry mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return (
            torch.stack(tuple(routes[name][0] for name in ROUTE_NAMES), dim=1),
            torch.stack(tuple(routes[name][1] for name in ROUTE_NAMES), dim=1),
        )

    def _feed_route_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if self.learned_retriever is None:
            query = self.query_embedding(requests, state)
            ann_item, ann_score = self.faiss.search(
                query, self.config.route_k * self.config.ann_oversample,
            )
        else:
            ann_item, ann_score = self.learned_retriever.retrieve(
                requests, state, self.config.route_k * self.config.ann_oversample,
            )
        ann_item, ann_score = self._filter_and_trim(
            requests,
            state,
            ann_item,
            ann_score,
            required_surface=Surface.FEED,
            allowed_lifecycle=(int(ContentLifecycle.RECENT),),
        )
        last_item = self._last_item(requests, state)
        graph_item = self.graph.neighbor[last_item.clamp_min(0)]
        graph_score = self.graph.score[last_item.clamp_min(0)]
        graph_item = torch.where(
            (last_item >= 0)[:, None], graph_item, torch.full_like(graph_item, -1)
        )
        graph_item, graph_score = self._filter_and_trim(
            requests,
            state,
            graph_item,
            graph_score,
            required_surface=Surface.FEED,
            allowed_lifecycle=(int(ContentLifecycle.RECENT),),
        )
        signals = build_feed_route_signals(
            self.catalog, state, requests.event_time.max(),
        )
        random_item, random_value = self._rotating_lifecycle_candidates(
            requests,
            state,
            signals.random,
            MAIN_FEED_LIFECYCLES,
        )
        popular_item, popular_value = self._top_for_surface(
            requests,
            state,
            signals.popular,
            Surface.FEED,
            allowed_lifecycle=tuple(
                int(value) for value in MAIN_FEED_LIFECYCLES
            ),
        )
        cold_item, cold_value = self._rotating_lifecycle_candidates(
            requests,
            state,
            signals.cold_start,
            ContentLifecycle.COLD_START,
        )
        hot_item, hot_value = self._top_for_surface(
            requests,
            state,
            signals.hot,
            Surface.FEED,
            allowed_lifecycle=(int(ContentLifecycle.HOT),),
        )
        evergreen_item, evergreen_value = self._top_for_surface(
            requests,
            state,
            signals.evergreen,
            Surface.FEED,
            allowed_lifecycle=(int(ContentLifecycle.EVERGREEN),),
        )
        followed_creator = self._last_followed_creator(requests, state)
        followed_creator = torch.where(
            requests.surface == int(Surface.FEED),
            followed_creator,
            torch.full_like(followed_creator, -1),
        )
        following_item, following_score = self._top_by_group(
            requests,
            state,
            signals.following,
            followed_creator,
            state.item_creator_id,
            allowed_lifecycle=(
                int(ContentLifecycle.COLD_START),
                int(ContentLifecycle.RECENT),
                int(ContentLifecycle.HOT),
                int(ContentLifecycle.EVERGREEN),
            ),
        )
        return {
            "random": (random_item, random_value),
            "popular": (popular_item, popular_value),
            "recent_ann": (ann_item, ann_score),
            "recent_graph": (graph_item, graph_score),
            "following": (following_item, following_score),
            "cold_start": (cold_item, cold_value),
            "hot": (hot_item, hot_value),
            "evergreen": (evergreen_item, evergreen_value),
        }

    def _business_route_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
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
        posting_region = torch.where(
            requests.surface == int(Surface.POSTING),
            region,
            torch.full_like(region, -1),
        )
        posting_item, posting_score = self._top_by_group(
            requests,
            state,
            self.catalog.quality_prior + 0.15 * torch.log1p(impression),
            posting_region,
            state.item_region,
        )
        commerce_item, commerce_score = self._top_for_surface(
            requests,
            state,
            0.45 * self.catalog.quality_prior
            + 0.35 * state.item_inventory
            + 0.20 * torch.log1p(state.item_bid),
            Surface.COMMERCE,
        )
        live_item, live_score = self._top_for_surface(
            requests,
            state,
            0.55 * self.catalog.quality_prior
            + 0.45 * engagement_rate,
            Surface.LIVE,
        )
        search_item, search_score = self._top_by_group(
            requests,
            state,
            self.catalog.quality_prior + 0.15 * popular_score,
            requests.query_topic,
            self.catalog.topic_id,
            require_query=True,
        )
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
            "posting_context": (posting_item, posting_score),
            "commerce_intent": (commerce_item, commerce_score),
            "live_now": (live_item, live_score),
            "search": (search_item, search_score),
            "retarget": (retarget_item, retarget_score),
        }

    def _surface_top(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        item = torch.full(
            (len(requests.user_id), self.config.route_k),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        values = torch.full_like(item, -torch.inf, dtype=torch.float)
        for surface in torch.unique(requests.surface).tolist():
            rows = requests.surface == surface
            eligible = state.item_active & surface_eligibility(
                surface, self.catalog.content_kind,
            )
            count = min(self.config.route_k, int(eligible.sum()))
            if not count:
                continue
            candidates = torch.where(eligible)[0]
            top = torch.topk(score[candidates], count).indices
            chosen = candidates[top]
            item[rows, :count] = chosen[None]
            values[rows, :count] = score[chosen][None]
        return item, values

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
