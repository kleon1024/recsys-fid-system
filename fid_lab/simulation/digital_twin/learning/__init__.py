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
from .publish_queue import (
    PUBLISH_QUEUE_VALUE_VERSION,
    load_publish_queue_batch,
    publish_queue_task_weights,
)
from .sparse_linear import (
    SparseLinearArtifact,
    train_sparse_linear,
)

__all__ = (
    "ArtifactCompatibility",
    "FACTUAL_REQUEST_STREAM_SCHEMA",
    "FactualRequestPartition",
    "FactualRequestPartitionRef",
    "FactualRequestStream",
    "Lane",
    "LaneCursor",
    "PartitionedSampleBus",
    "PUBLISH_QUEUE_VALUE_VERSION",
    "PersistentModelRegistry",
    "ProbeArtifact",
    "ProbeBatch",
    "SparseLinearArtifact",
    "feature_drift_report",
    "load_probe_batch",
    "load_publish_queue_batch",
    "publish_queue_task_weights",
    "train_probe",
    "train_sparse_linear",
)
