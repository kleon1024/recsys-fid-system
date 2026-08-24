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
from .world import UserEcosystemWorld, UserWorldConfig, UserWorldSnapshot

__all__ = (
    "AppEventBatch",
    "AtomicSimulationKernel",
    "ContentKind",
    "EventType",
    "ExperimentAssignment",
    "ExperimentPlan",
    "ObservableEventLog",
    "PlatformRequestBatch",
    "PublicCatalog",
    "RenderedSlateBatch",
    "Surface",
    "TickResult",
    "UserEcosystemWorld",
    "UserWorldConfig",
    "UserWorldSnapshot",
    "build_public_catalog",
    "make_app_events",
)
