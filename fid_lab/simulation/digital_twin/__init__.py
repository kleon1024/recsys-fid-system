"""Causal, event-driven recommendation ecosystem simulator."""

from .contracts import (
    APP_EVENT_SCHEMA_VERSION,
    AppEventBatch,
    ContentKind,
    EventType,
    PlatformRequestBatch,
    PublishFailureReason,
    RenderedSlateBatch,
    Surface,
    make_app_events,
)
from .catalog import PublicCatalog, build_public_catalog
from .engine import (
    AtomicSimulationKernel,
    ExperimentAssignment,
    ExperimentPlan,
    LayerAssignmentTrace,
    TickResult,
)
from .event_log import ObservableEventLog
from .observability import (
    CheckpointRecord,
    FullFlowSnapshot,
    build_full_flow_tables,
    materialize_full_flow,
)
from .experiments.layered import LayeredExperimentPlan, PolicyLayer
from .platform import (
    CascadePolicy,
    ContentLifecycle,
    LifecycleConfig,
    ObservableProjection,
    ProjectionSnapshot,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
)
from .samples.contracts import (
    RequestCandidateTrace,
    ServingOutput,
    TraceManifest,
)
from .samples.joiner import (
    JoinerConfig,
    RequestLevelJoiner,
    capture_request_context,
)
from .samples.negative_sampling import (
    NegativeSource,
    corrected_sampled_softmax_loss,
    negative_source_counts,
)
from .world import (
    DelayedOutcomeQueue,
    HiddenSupplyState,
    SupplyEcosystem,
    UserEcosystemWorld,
    UserWorldConfig,
    UserWorldSnapshot,
)

__all__ = (
    "AppEventBatch",
    "APP_EVENT_SCHEMA_VERSION",
    "AtomicSimulationKernel",
    "CascadePolicy",
    "ContentLifecycle",
    "CheckpointRecord",
    "ContentKind",
    "DelayedOutcomeQueue",
    "EventType",
    "ExperimentAssignment",
    "ExperimentPlan",
    "FullFlowSnapshot",
    "LayerAssignmentTrace",
    "LayeredExperimentPlan",
    "HiddenSupplyState",
    "JoinerConfig",
    "LifecycleConfig",
    "NegativeSource",
    "ObservableEventLog",
    "ObservableProjection",
    "PlatformRequestBatch",
    "PolicyLayer",
    "PublishFailureReason",
    "PublicCatalog",
    "ProjectionSnapshot",
    "RequestCandidateTrace",
    "RequestLevelJoiner",
    "RankingConfig",
    "ReferencePlatformConfig",
    "ReferenceRecommendationPlatform",
    "RenderedSlateBatch",
    "ServingOutput",
    "Surface",
    "SupplyEcosystem",
    "TickResult",
    "TraceManifest",
    "RetrievalConfig",
    "UserEcosystemWorld",
    "UserWorldConfig",
    "UserWorldSnapshot",
    "build_public_catalog",
    "build_full_flow_tables",
    "capture_request_context",
    "corrected_sampled_softmax_loss",
    "make_app_events",
    "materialize_full_flow",
    "negative_source_counts",
)
