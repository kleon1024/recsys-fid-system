"""Contracts for the device-resident Feed simulator."""

from __future__ import annotations

from dataclasses import dataclass

from ....simulation.contracts import DEFAULT_SEARCH_EVENT_RATE


DEFAULT_GPU_BATCH_USERS = 200_000
CANDIDATE_GRAPH_VERSION = "multiroute-rrf-cascade-v3"
EXTERNAL_MIXTURE_FEED_VERSION = "external-sequence-mixture-v4"


@dataclass(frozen=True)
class TensorFeedConfig:
    users: int = 1_000_000
    steps: int = 24
    candidates: int = 20
    route_candidates: int = 8
    route_oversample: int = 3
    merged_candidates: int = 48
    audit_candidates: int = 24
    candidate_graph_version: str = CANDIDATE_GRAPH_VERSION
    trace_users: int = 0
    trace_requests_per_user: int = 2
    topics: int = 12
    catalog_items: int = 200_000
    catalog_creators: int | None = None
    batch_users: int = DEFAULT_GPU_BATCH_USERS
    seed: int = 20260823
    catalog_seed: int | None = None
    retain_paired_user_metrics: bool = False
    device: str = "cuda:0"
    count_inactive_play_bug: bool = False
    signal_version: str = "industrial-cross-sequence-v1"
    max_sessions: int = 4
    requests_per_session: int = 8
    search_event_rate: float = DEFAULT_SEARCH_EVENT_RATE
    search_ttl_requests: int = 3
    behavior_sequence_length: int = 64

    def __post_init__(self) -> None:
        if self.catalog_creators is None:
            object.__setattr__(
                self, "catalog_creators", min(25_000, self.catalog_items)
            )
        if self.signal_version not in {
            "industrial-cross-sequence-v1",
            "heterogeneous-nonlinear-v2",
            "kuairand-calibrated-v3",
            "kuairand-local-neural-v4",
            EXTERNAL_MIXTURE_FEED_VERSION,
        }:
            raise ValueError(f"unsupported signal version: {self.signal_version}")
        if not 0.0 <= self.search_event_rate <= 1.0:
            raise ValueError("search event rate must be in [0, 1]")
        if self.search_ttl_requests < 1:
            raise ValueError("search TTL must be positive")
        if self.route_candidates < 1 or self.route_oversample < 1:
            raise ValueError("route candidate budgets must be positive")
        if self.merged_candidates < self.candidates:
            raise ValueError("merged candidates must cover the coarse output")
        if self.audit_candidates < 1:
            raise ValueError("audit candidate count must be positive")
        if not 1 <= self.catalog_creators <= self.catalog_items:
            raise ValueError("catalog creators must fit inside the item corpus")
        if self.candidate_graph_version != CANDIDATE_GRAPH_VERSION:
            raise ValueError("unsupported candidate graph version")
        if self.trace_users < 0 or self.trace_requests_per_user < 1:
            raise ValueError("trace sampling limits are invalid")
        if self.behavior_sequence_length < 8:
            raise ValueError("Feed behavior sequence must contain at least eight events")
