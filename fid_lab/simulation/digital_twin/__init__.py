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
    TickResult,
)
from .event_log import ObservableEventLog
from .platform import ObservableProjection, ProjectionSnapshot
from .samples import (
    JoinerConfig,
    RequestCandidateTrace,
    RequestLevelJoiner,
    TraceManifest,
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
    "ContentKind",
    "DelayedOutcomeQueue",
    "EventType",
    "ExperimentAssignment",
    "ExperimentPlan",
    "HiddenSupplyState",
    "JoinerConfig",
    "ObservableEventLog",
    "ObservableProjection",
    "PlatformRequestBatch",
    "PublicCatalog",
    "ProjectionSnapshot",
    "RequestCandidateTrace",
    "RequestLevelJoiner",
    "RenderedSlateBatch",
    "Surface",
    "SupplyEcosystem",
    "TickResult",
    "TraceManifest",
    "UserEcosystemWorld",
    "UserWorldConfig",
    "UserWorldSnapshot",
    "build_public_catalog",
    "capture_request_context",
    "make_app_events",
)
