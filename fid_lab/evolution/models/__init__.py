"""Thin adapters around mature OSS recommendation components."""

from .deepctr_adapter import DeepCTRModelAdapter, supported_deepctr_models
from .esmm import ESMM, ESMMOutput
from .retrieval import MultiInterestTwoTower, RetrievalSnapshot, TwoTowerRetriever

__all__ = [
    "DeepCTRModelAdapter",
    "ESMM",
    "ESMMOutput",
    "MultiInterestTwoTower",
    "RetrievalSnapshot",
    "TwoTowerRetriever",
    "supported_deepctr_models",
]
