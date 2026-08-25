"""Persistent v4 sample consumption, probe training and model lifecycle."""

from .contracts import ArtifactCompatibility, Lane, LaneCursor, ProbeBatch
from .probe import (
    ProbeArtifact,
    feature_drift_report,
    load_probe_batch,
    train_probe,
)
from .registry import PersistentModelRegistry
from .request_stream import (
    FACTUAL_REQUEST_STREAM_SCHEMA,
    FactualRequestPartition,
    FactualRequestPartitionRef,
    FactualRequestStream,
)
from .sample_bus import PartitionedSampleBus

__all__ = (
    "ArtifactCompatibility",
    "FACTUAL_REQUEST_STREAM_SCHEMA",
    "FactualRequestPartition",
    "FactualRequestPartitionRef",
    "FactualRequestStream",
    "Lane",
    "LaneCursor",
    "PartitionedSampleBus",
    "PersistentModelRegistry",
    "ProbeArtifact",
    "ProbeBatch",
    "feature_drift_report",
    "load_probe_batch",
    "train_probe",
)
