"""KuaiRand-backed external sequence evidence for the V4 research lane."""

from .data import (
    build_randomized_dataset,
    build_sequence_dataset,
    load_randomized_split,
    load_sequence_split,
)
from .kernel import KuaiBehaviorKernel

__all__ = [
    "KuaiBehaviorKernel",
    "build_sequence_dataset",
    "build_randomized_dataset",
    "load_randomized_split",
    "load_sequence_split",
]
