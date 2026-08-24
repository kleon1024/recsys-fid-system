"""Causal, event-driven recommendation ecosystem simulator."""

from .contracts import (
    AppEventBatch,
    ContentKind,
    EventType,
    PlatformRequestBatch,
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
    "AtomicSimulationKernel",
    "CascadePolicy",
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
    "ObservableEventLog",
    "ObservableProjection",
    "PlatformRequestBatch",
    "PolicyLayer",
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
    "make_app_events",
    "materialize_full_flow",
)
