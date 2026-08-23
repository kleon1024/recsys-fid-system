"""Thin adapters around mature OSS recommendation components."""

from .deepctr_adapter import DeepCTRModelAdapter, supported_deepctr_models
from .retrieval import MultiInterestTwoTower, RetrievalSnapshot, TwoTowerRetriever

__all__ = [
    "DeepCTRModelAdapter",
    "MultiInterestTwoTower",
    "RetrievalSnapshot",
    "TwoTowerRetriever",
    "supported_deepctr_models",
]
