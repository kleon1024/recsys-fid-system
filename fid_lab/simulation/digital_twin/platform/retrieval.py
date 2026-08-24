"""Independent observable retrieval routes with FAISS and sparse co-visits."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys

import numpy as np
import scipy.sparse
import torch

from ..catalog import PublicCatalog
from ..contracts import AppEventBatch, ContentKind, EventType, PlatformRequestBatch
from ..contracts import Surface
from .projection import ITEM_COUNTER_EVENTS, PlatformProjectionState


ROUTE_NAMES = (
    "ann",
    "graph",
    "geo",
    "fresh",
    "long_tail",
    "popular",
    "search",
    "retarget",
)

SURFACE_CONTENT = {
    Surface.FEED: (
        ContentKind.SHORT_VIDEO,
        ContentKind.PHOTO,
        ContentKind.ARTICLE,
        ContentKind.CARD,
        ContentKind.LIVE_ROOM,
        ContentKind.PRODUCT,
        ContentKind.POI,
        ContentKind.AD,
    ),
    Surface.SEARCH: (
        ContentKind.SHORT_VIDEO,
        ContentKind.PHOTO,
        ContentKind.ARTICLE,
        ContentKind.CARD,
        ContentKind.PRODUCT,
        ContentKind.POI,
    ),
    Surface.COMMERCE: (
        ContentKind.PRODUCT,
        ContentKind.LIVE_ROOM,
        ContentKind.AD,
    ),
    Surface.LIVE: (ContentKind.LIVE_ROOM,),
    Surface.LOCAL: (
        ContentKind.SHORT_VIDEO,
        ContentKind.PHOTO,
        ContentKind.CARD,
        ContentKind.PRODUCT,
        ContentKind.POI,
    ),
    Surface.POSTING: (
        ContentKind.POI,
        ContentKind.PRODUCT,
        ContentKind.CREATOR_PROMPT,
    ),
}


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


def surface_eligibility(
    surface: int | torch.Tensor,
    content_kind: torch.Tensor,
) -> torch.Tensor:
    if isinstance(surface, int):
        allowed = SURFACE_CONTENT[Surface(surface)]
        result = torch.zeros_like(content_kind, dtype=torch.bool)
        for kind in allowed:
            result |= content_kind == int(kind)
        return result
    result = torch.zeros(
        len(surface), content_kind.shape[-1],
        device=content_kind.device,
        dtype=torch.bool,
    )
    for candidate_surface, allowed in SURFACE_CONTENT.items():
        rows = surface == int(candidate_surface)
        if not rows.any():
            continue
        kinds = content_kind[rows]
        for kind in allowed:
            result[rows] |= kinds == int(kind)
    return result


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
        if (self._indexed_active & ~active).any():
            raise ValueError("ANN item deletion requires an explicit index rebuild")
        if self.backend == "torch":
            self._torch_item = self.catalog.item_id[active]
            self._torch_embedding = self.catalog.content_embedding[active]
            self._indexed_active = active.clone()
        else:
            faiss = importlib.import_module("faiss")
            faiss.omp_set_num_threads(int(os.environ.get("FID_FAISS_THREADS", "1")))
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

    def _top_by_group(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        score: torch.Tensor,
        group: torch.Tensor,
        item_group: torch.Tensor,
        *,
        require_query: bool = False,
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

    def retrieve(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        enabled_routes: tuple[str, ...] = ROUTE_NAMES,
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
        enabled = torch.tensor(
            [name in enabled_routes for name in ROUTE_NAMES],
            device=self.device,
            dtype=torch.bool,
        )
        route_valid &= enabled[None, :, None]
        merged_item, merged_score, route_bits = self._rrf(
            route_item, route_valid,
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
            index_version=self.faiss.version,
        )

    def _route_candidates(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_embedding(requests, state)
        ann_item, ann_score = self.faiss.search(
            query, self.config.route_k * self.config.ann_oversample,
        )
        ann_item, ann_score = self._filter_and_trim(
            requests, state, ann_item, ann_score,
        )
        last_item = self._last_item(requests, state)
        graph_item = self.graph.neighbor[last_item.clamp_min(0)]
        graph_score = self.graph.score[last_item.clamp_min(0)]
        graph_item = torch.where(
            (last_item >= 0)[:, None], graph_item, torch.full_like(graph_item, -1)
        )
        graph_item, graph_score = self._filter_and_trim(
            requests, state, graph_item, graph_score,
        )
        impression = state.item_event_counts[
            :, ITEM_COUNTER_EVENTS.index(EventType.IMPRESSION)
        ]
        engagement = state.item_event_counts[
            :, ITEM_COUNTER_EVENTS.index(EventType.LONG_VIEW)
        ] + state.item_event_counts[:, ITEM_COUNTER_EVENTS.index(EventType.CLICK)]
        popular_score = torch.log1p(impression) + 0.4 * torch.log1p(engagement)
        popular_score += 0.25 * self.catalog.quality_prior
        fresh_score = (
            -0.002 * (
                requests.event_time.max().float()
                - state.item_publish_time.clamp_max(requests.event_time.max()).float()
            )
            + 0.35 * self.catalog.quality_prior
        )
        tail_score = (
            0.65 * self.catalog.quality_prior - 0.18 * torch.log1p(impression)
        )
        popular_item, popular_value = self._surface_top(
            requests, state, popular_score,
        )
        fresh_item, fresh_value = self._surface_top(
            requests, state, fresh_score,
        )
        tail_item, tail_value = self._surface_top(
            requests, state, tail_score,
        )
        region = state.user_region[requests.user_id]
        geo_item, geo_score = self._top_by_group(
            requests,
            state,
            self.catalog.quality_prior + 0.2 * state.item_inventory,
            region,
            self.catalog.region,
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
        item = torch.stack((
            ann_item,
            graph_item,
            geo_item,
            fresh_item,
            tail_item,
            popular_item,
            search_item,
            retarget_item,
        ), dim=1)
        score = torch.stack((
            ann_score,
            graph_score,
            geo_score,
            fresh_value,
            tail_value,
            popular_value,
            search_score,
            retarget_score,
        ), dim=1)
        return item, score

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = item.clamp_min(0)
        valid = item >= 0
        valid &= state.item_active[safe]
        valid &= surface_eligibility(
            requests.surface,
            self.catalog.content_kind[safe],
        )
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
        return chosen_item, chosen_score

    def _rrf(
        self, route_item: torch.Tensor, route_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requests, routes, route_k = route_item.shape
        rank = torch.arange(1, route_k + 1, device=self.device).float()
        contribution = 1.0 / (
            self.config.reciprocal_rank_constant + rank
        )
        score = contribution[None, None].expand(requests, routes, route_k)
        score = score * route_valid.float()
        bit = (
            2 ** torch.arange(routes, device=self.device, dtype=torch.long)
        )[None, :, None].expand_as(route_item) * route_valid.long()
        flat_item = torch.where(
            route_valid, route_item, torch.full_like(route_item, torch.iinfo(torch.long).max)
        ).reshape(requests, -1)
        flat_score = score.reshape(requests, -1)
        flat_bit = bit.reshape(requests, -1)
        ordered_item, order = torch.sort(flat_item, dim=1)
        ordered_score = torch.gather(flat_score, 1, order)
        ordered_bit = torch.gather(flat_bit, 1, order)
        starts = torch.ones_like(ordered_item, dtype=torch.bool)
        starts[:, 1:] = ordered_item[:, 1:] != ordered_item[:, :-1]
        group = torch.cumsum(starts.long(), dim=1) - 1
        width = ordered_item.shape[1]
        merged_item = torch.full_like(ordered_item, -1)
        merged_score = torch.zeros(requests, width, device=self.device)
        merged_bit = torch.zeros_like(ordered_item)
        merged_item.scatter_(1, group, ordered_item)
        merged_score.scatter_add_(1, group, ordered_score)
        merged_bit.scatter_add_(1, group, ordered_bit)
        valid = merged_item != torch.iinfo(torch.long).max
        merged_score.masked_fill_(~valid, -torch.inf)
        keep = min(self.config.merged_k, width)
        position = torch.topk(merged_score, keep, dim=1).indices
        item = torch.gather(merged_item, 1, position)
        value = torch.gather(merged_score, 1, position)
        bits = torch.gather(merged_bit, 1, position)
        item = torch.where(torch.isfinite(value), item, torch.full_like(item, -1))
        if keep < self.config.merged_k:
            padding = self.config.merged_k - keep
            item = torch.nn.functional.pad(item, (0, padding), value=-1)
            value = torch.nn.functional.pad(value, (0, padding), value=-torch.inf)
            bits = torch.nn.functional.pad(bits, (0, padding), value=0)
        return item, value, bits
