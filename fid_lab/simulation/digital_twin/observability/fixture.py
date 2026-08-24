"""Small deterministic full-flow partition for analytical acceptance."""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import build_public_catalog
from ..contracts import AppEventBatch
from ..contracts import Surface
from ..engine import AtomicSimulationKernel, ExperimentPlan
from ..event_log import ObservableEventLog
from ..platform import (
    CascadePolicy,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
)
from ..samples.joiner import JoinerConfig, RequestLevelJoiner
from ..world import UserEcosystemWorld, UserWorldConfig
from .contracts import CheckpointRecord, FullFlowSnapshot


@dataclass(frozen=True)
class FullFlowFixtureConfig:
    users: int = 64
    items: int = 600
    platform_seed: int = 2_001
    environment_seed: int = 2_003
    experiment_seed: int = 2_009
    device: str = "cpu"
    route_k: int = 6
    merged_k: int = 24
    coarse_k: int = 12
    fine_k: int = 6
    expose_k: int = 3
    history_length: int = 8
    recall_negatives: int = 4
    logical_time: int = 0
    scenario: str = "mixed"

    def __post_init__(self):
        dimensions = (
            self.users,
            self.items,
            self.route_k,
            self.merged_k,
            self.coarse_k,
            self.fine_k,
            self.expose_k,
            self.history_length,
            self.recall_negatives,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("fixture dimensions must be positive")
        if self.logical_time < 0:
            raise ValueError("fixture logical time cannot be negative")
        if self.scenario not in {"mixed", "feed_posting_cycle"}:
            raise ValueError("fixture scenario is unsupported")
        if not self.merged_k >= self.coarse_k >= self.fine_k >= self.expose_k:
            raise ValueError("fixture cascade budgets are inconsistent")


def build_full_flow_fixture(
    config: FullFlowFixtureConfig = FullFlowFixtureConfig(),
) -> FullFlowSnapshot:
    return build_full_flow_fixtures(config, ticks=1)[0]


def build_full_flow_fixtures(
    config: FullFlowFixtureConfig = FullFlowFixtureConfig(),
    *,
    ticks: int,
) -> tuple[FullFlowSnapshot, ...]:
    if ticks <= 0:
        raise ValueError("fixture ticks must be positive")
    catalog = build_public_catalog(
        items=config.items,
        creators=max(config.items // 15, 1),
        merchants=max(config.items // 30, 1),
        advertisers=max(config.items // 60, 1),
        topics=8,
        countries=2,
        regions_per_country=3,
        embedding_dim=8,
        platform_seed=config.platform_seed,
        device=config.device,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=config.users,
        topics=8,
        embedding_dim=8,
        countries=2,
        regions_per_country=3,
        environment_seed=config.environment_seed,
        future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(
            users=config.users,
            history_length=config.history_length,
        ),
        catalog,
        RetrievalConfig(
            route_k=config.route_k,
            merged_k=config.merged_k,
            graph_neighbors=config.route_k,
        ),
        RankingConfig(
            coarse_k=config.coarse_k,
            fine_k=config.fine_k,
            expose_k=config.expose_k,
        ),
    )
    event_log = ObservableEventLog(allowed_lateness=world.max_reporting_lag)
    experiment = ExperimentPlan.ramped_user_ab(
        active_policy=CascadePolicy("active", 1, 1, 1),
        treatment_policy=CascadePolicy("candidate", 2, 2, 2),
        experiment_seed=config.experiment_seed,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    kernel = AtomicSimulationKernel(world, platform, event_log)
    snapshots = []
    final_time = config.logical_time + ticks
    for logical_time in range(final_time):
        if config.scenario == "feed_posting_cycle":
            world.users.surface_intent.fill_(1e-8)
            world.users.surface_intent[:, int(
                Surface.POSTING if logical_time == 0 else Surface.FEED
            )] = 1.0
            world.users.habit.fill_(1.0)
        result = kernel.step(logical_time, experiment)
        if logical_time < config.logical_time:
            continue
        if result.candidate_trace is None or result.request_context is None:
            raise RuntimeError("fixture cascade did not emit full-flow traces")
        events = AppEventBatch.concatenate((
            result.entry_events,
            result.response_events,
        ))
        samples = RequestLevelJoiner(
            JoinerConfig(
                ticks_per_day=96,
                recall_negatives=config.recall_negatives,
            ), catalog,
        ).materialize(
            result.candidate_trace,
            result.request_context,
            events,
            event_watermark=logical_time,
        )
        snapshots.append(FullFlowSnapshot(
            catalog=catalog,
            trace=result.candidate_trace,
            context=result.request_context,
            events=events,
            samples=samples,
            projection=platform.projection.snapshot(),
            checkpoints=(CheckpointRecord(
                created_time=logical_time,
                lane="active",
                model_name="reference-cascade",
                checkpoint_version="checkpoint-v1",
                data_watermark=logical_time,
                sample_manifest="fixture-samples-v1",
                feature_version=result.candidate_trace.manifest.feature_version,
                fid_version=result.candidate_trace.manifest.fid_version,
                index_version=result.candidate_trace.manifest.index_version,
                validation_status="pass",
                publish_state="active",
            ),),
            layer_assignment=result.layer_assignment,
        ))
    return tuple(snapshots)
