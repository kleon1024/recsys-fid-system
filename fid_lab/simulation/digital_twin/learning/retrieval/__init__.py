"""P3-06 retrieval training, artifacts and fixed-budget evaluation."""

from .artifact import LearnedRetrievalAdapter, RetrievalANNIndex, RetrievalArtifact
from .contracts import RetrievalCorpus, RetrievalModelConfig, RetrievalQueryBatch
from .data import corpus_from_snapshot, load_retrieval_batch
from .model import ObservableRetrievalModel
from .training import train_retrieval_model

__all__ = [
    "LearnedRetrievalAdapter",
    "ObservableRetrievalModel",
    "RetrievalANNIndex",
    "RetrievalArtifact",
    "RetrievalCorpus",
    "RetrievalModelConfig",
    "RetrievalQueryBatch",
    "corpus_from_snapshot",
    "load_retrieval_batch",
    "train_retrieval_model",
]
