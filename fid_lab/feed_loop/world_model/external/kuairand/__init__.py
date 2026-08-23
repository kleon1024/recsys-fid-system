"""KuaiRand-backed external sequence evidence for the V4 research lane."""

from .dataset import build_sequence_dataset, load_sequence_split
from .kernel import KuaiBehaviorKernel

__all__ = [
    "KuaiBehaviorKernel",
    "build_sequence_dataset",
    "load_sequence_split",
]
