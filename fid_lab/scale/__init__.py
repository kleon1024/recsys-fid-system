"""Large-scale synthetic distribution, tensor, and experiment contracts."""

from .contracts import FEED_TASKS, ScaleConfig
from .dataset import FeedTensorDataset, TensorSchema, tensor_schema
from .diagnostics import ExperimentDiagnostic, diagnose_auc_without_lift
from .synthetic import ScaleDataset, build_scale_dataset, summarize_distribution

__all__ = [
    "ExperimentDiagnostic",
    "FEED_TASKS",
    "FeedTensorDataset",
    "ScaleConfig",
    "ScaleDataset",
    "TensorSchema",
    "build_scale_dataset",
    "diagnose_auc_without_lift",
    "summarize_distribution",
    "tensor_schema",
]
