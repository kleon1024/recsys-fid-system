"""Index configuration and learned-retriever serving contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ...contracts import PlatformRequestBatch
from ..projection import PlatformProjectionState


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
    selection_seed: int = 2_029
    popular_pool_multiplier: int = 16
    popular_interest_fraction: float = 0.25
    interest_half_life_ticks: int = 96

    def __post_init__(self):
        dimensions = (
            self.route_k, self.merged_k, self.ann_oversample,
            self.graph_neighbors, self.refresh_interval,
            self.hnsw_neighbors, self.hnsw_ef_search,
            self.popular_pool_multiplier,
            self.interest_half_life_ticks,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("retrieval dimensions must be positive")
        if not 0.0 < self.popular_interest_fraction <= 1.0:
            raise ValueError("Popular interest fraction must be in (0, 1]")


class LearnedRetriever(Protocol):
    serving_version_id: int

    @property
    def index_version(self) -> str: ...

    def retrieve(
        self, requests: PlatformRequestBatch,
        state: PlatformProjectionState, top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
