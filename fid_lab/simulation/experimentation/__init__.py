"""Orthogonal experiment assignment and full-chain parameter snapshots."""

from .assignment import (
    assign_binary_torch,
    assign_layer_numpy,
    assign_layers,
    validate_layer_ownership,
)
from .contracts import Experiment, ExperimentLayer, FeedParameters, Variant

__all__ = [
    "Experiment",
    "ExperimentLayer",
    "FeedParameters",
    "Variant",
    "assign_layers",
    "assign_layer_numpy",
    "validate_layer_ownership",
]
