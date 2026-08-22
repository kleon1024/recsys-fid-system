"""Thin adapters around mature OSS recommendation components."""

from .deepctr_adapter import DeepCTRModelAdapter, supported_deepctr_models
from .retrieval import MultiInterestTwoTower, TwoTowerRetriever

__all__ = [
    "DeepCTRModelAdapter",
    "MultiInterestTwoTower",
    "TwoTowerRetriever",
    "supported_deepctr_models",
]
