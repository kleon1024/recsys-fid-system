"""Contracts for the device-resident Feed simulator."""

from __future__ import annotations

from dataclasses import dataclass

from ....simulation.contracts import DEFAULT_SEARCH_EVENT_RATE


DEFAULT_GPU_BATCH_USERS = 200_000


@dataclass(frozen=True)
class TensorFeedConfig:
    users: int = 1_000_000
    steps: int = 24
    candidates: int = 20
    route_candidates: int = 8
    route_oversample: int = 3
    merged_candidates: int = 48
    audit_candidates: int = 24
    candidate_graph_version: str = "multiroute-rrf-coarse-v2"
    trace_users: int = 0
    trace_requests_per_user: int = 2
    topics: int = 12
    catalog_items: int = 200_000
    batch_users: int = DEFAULT_GPU_BATCH_USERS
    seed: int = 20260823
    device: str = "cuda:0"
    count_inactive_play_bug: bool = False
    signal_version: str = "industrial-cross-sequence-v1"
    max_sessions: int = 4
    requests_per_session: int = 8
    search_event_rate: float = DEFAULT_SEARCH_EVENT_RATE
    search_ttl_requests: int = 3

    def __post_init__(self) -> None:
        if self.signal_version not in {
            "industrial-cross-sequence-v1",
            "heterogeneous-nonlinear-v2",
            "kuairand-calibrated-v3",
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
        if self.candidate_graph_version != "multiroute-rrf-coarse-v2":
            raise ValueError("unsupported candidate graph version")
        if self.trace_users < 0 or self.trace_requests_per_user < 1:
            raise ValueError("trace sampling limits are invalid")


