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
from .world import (
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
    "EventType",
    "ExperimentAssignment",
    "ExperimentPlan",
    "HiddenSupplyState",
    "ObservableEventLog",
    "PlatformRequestBatch",
    "PublicCatalog",
    "RenderedSlateBatch",
    "Surface",
    "SupplyEcosystem",
    "TickResult",
    "UserEcosystemWorld",
    "UserWorldConfig",
    "UserWorldSnapshot",
    "build_public_catalog",
    "make_app_events",
)
