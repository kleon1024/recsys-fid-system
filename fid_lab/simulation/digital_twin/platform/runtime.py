"""Projection-backed reference cascade that emits complete serving traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    PlatformRequestBatch,
    RenderedSlateBatch,
)
from ..samples.contracts import (
    RequestCandidateTrace,
    ServingOutput,
    TraceManifest,
)
from ..samples.joiner import capture_request_context
from .projection import ObservableProjection, ProjectionSnapshot
from .lifecycle import LIFECYCLE_POLICY_VERSION, LifecycleConfig
from .ranking import CascadePolicy, CascadeRanker, RankingConfig
from .requests import open_platform_requests
from .retrieval import MultiRouteRetriever, RetrievalConfig


@dataclass(frozen=True)
class ReferencePlatformConfig:
    users: int
    history_length: int = 128
    recall_version_id: int = 1
    feature_version: str = "observable-features-v1"
    catalog_version: str = "public-catalog-v1"
    policy_registry_version: str = "reference-policy-registry-v1"
    fid_version: str = "fid-v2"
    ticks_per_day: int = 96

    def __post_init__(self):
        if self.users <= 0 or self.history_length <= 0 or self.ticks_per_day <= 0:
            raise ValueError("reference platform dimensions must be positive")


@dataclass(frozen=True)
class PlatformServingSnapshot:
    projection: ProjectionSnapshot
    index_version: str


class ReferenceRecommendationPlatform:
    """Observable-only cascade; intended as the v4 serving integration path."""

    def __init__(
        self,
        config: ReferencePlatformConfig,
        catalog: PublicCatalog,
        retrieval_config: RetrievalConfig | None = None,
        ranking_config: RankingConfig | None = None,
    ):
        self.config = config
        self.catalog = catalog
        self.projection = ObservableProjection(
            config.users,
            catalog,
            config.history_length,
            LifecycleConfig(ticks_per_day=config.ticks_per_day),
        )
        self.retriever = MultiRouteRetriever(
            catalog, retrieval_config or RetrievalConfig(),
        )
        self.ranker = CascadeRanker(
            catalog, self.retriever, ranking_config or RankingConfig(),
        )

    def ingest(self, events: AppEventBatch) -> None:
        self.retriever.ingest(events)
        self.projection.ingest(events)
        if len(events.ingest_time):
            self.retriever.refresh(
                self.projection.state, int(events.ingest_time.max()),
            )

    def snapshot(self) -> PlatformServingSnapshot:
        return PlatformServingSnapshot(
            projection=self.projection.view(),
            index_version=self.retriever.faiss.version,
        )

    def open_requests(
        self, entry_events: AppEventBatch,
    ) -> PlatformRequestBatch:
        return open_platform_requests(entry_events)

    def render(
        self,
        snapshot: PlatformServingSnapshot,
        requests: PlatformRequestBatch,
        policy: object,
        experiment_cell: int,
        assignment_probability: torch.Tensor,
    ) -> ServingOutput:
        if not isinstance(policy, CascadePolicy):
            raise TypeError("reference platform requires a CascadePolicy")
        retrieval = self.retriever.retrieve(
            requests, snapshot.projection.state, policy.enabled_routes,
        )
        ranked = self.ranker.rank(
            requests, snapshot.projection.state, retrieval, policy,
        )
        valid = ranked.exposed_item_id >= 0
        route_lifecycle = torch.where(
            retrieval.route_valid,
            snapshot.projection.state.item_lifecycle[
                retrieval.route_item_id.clamp_min(0)
            ],
            torch.full_like(retrieval.route_item_id, -1),
        )
        recall_lifecycle = torch.where(
            retrieval.item_id >= 0,
            snapshot.projection.state.item_lifecycle[
                retrieval.item_id.clamp_min(0)
            ],
            torch.full_like(retrieval.item_id, -1),
        )
        exposure_probability = valid.float()
        slate = RenderedSlateBatch(
            request_id=requests.request_id,
            user_id=requests.user_id,
            surface=requests.surface,
            event_time=requests.event_time,
            item_ids=ranked.exposed_item_id,
            positions=ranked.exposed_position,
            valid=valid,
            ui_variant=torch.full_like(requests.user_id, experiment_cell),
            exposure_probability=exposure_probability,
            assignment_probability=assignment_probability,
        )
        recall_version = torch.full_like(
            requests.user_id, policy.recall_version_id,
        )
        trace = RequestCandidateTrace(
            request_id=requests.request_id,
            user_id=requests.user_id,
            surface=requests.surface,
            event_time=requests.event_time,
            query_topic=requests.query_topic,
            user_country=snapshot.projection.state.user_country[requests.user_id],
            user_region=snapshot.projection.state.user_region[requests.user_id],
            user_creator_id=snapshot.projection.state.user_creator_id[
                requests.user_id
            ],
            route_item_id=retrieval.route_item_id,
            route_score=retrieval.route_score,
            route_valid=retrieval.route_valid,
            route_lifecycle_id=route_lifecycle,
            recall_item_id=retrieval.item_id,
            recall_route_id=retrieval.route_bits,
            recall_score=retrieval.score,
            recall_sampling_probability=retrieval.sampling_probability,
            recall_lifecycle_id=recall_lifecycle,
            coarse_item_id=ranked.coarse_item_id,
            coarse_score=ranked.coarse_score,
            coarse_sampling_probability=torch.where(
                ranked.coarse_item_id >= 0,
                torch.ones_like(ranked.coarse_score),
                torch.zeros_like(ranked.coarse_score),
            ),
            fine_item_id=ranked.fine_item_id,
            fine_score=ranked.fine_score,
            exposed_item_id=ranked.exposed_item_id,
            exposed_position=ranked.exposed_position,
            exposure_probability=exposure_probability,
            experiment_cell=torch.full_like(requests.user_id, experiment_cell),
            assignment_probability=assignment_probability,
            recall_version_id=recall_version,
            coarse_version_id=torch.full_like(
                requests.user_id, policy.coarse_version_id,
            ),
            fine_version_id=torch.full_like(
                requests.user_id, policy.fine_version_id,
            ),
            mix_version_id=torch.full_like(
                requests.user_id, policy.mix_version_id,
            ),
            manifest=TraceManifest(
                schema_version="request-candidate-trace-v2",
                feature_version=self.config.feature_version,
                catalog_version=self.config.catalog_version,
                policy_registry_version=self.config.policy_registry_version,
                route_names=self.retriever.route_names,
                index_version=snapshot.index_version,
                fid_version=self.config.fid_version,
                lifecycle_version=LIFECYCLE_POLICY_VERSION,
            ),
        )
        context = capture_request_context(trace, snapshot.projection)
        return ServingOutput(slate, trace, context)
