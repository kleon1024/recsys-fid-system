"""Persistent v4 sample consumption, probe training and model lifecycle."""

from .contracts import ArtifactCompatibility, Lane, LaneCursor, ProbeBatch
from .probe import (
    ProbeArtifact,
    feature_drift_report,
    load_probe_batch,
    train_probe,
)
from .registry import PersistentModelRegistry
from .sample_bus import PartitionedSampleBus

__all__ = (
    "ArtifactCompatibility",
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
