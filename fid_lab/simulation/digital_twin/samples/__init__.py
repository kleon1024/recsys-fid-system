"""Request-level sample authorities for the causal digital twin."""

from .contracts import (
    CoarseRankExampleBatch,
    FineRankExampleBatch,
    JoinedSampleAuthorities,
    RecallExampleBatch,
    RequestCandidateTrace,
    RequestContextBatch,
    ServingOutput,
    TraceManifest,
)
from .negative_sampling import (
    NegativeSource,
    corrected_sampled_softmax_loss,
    negative_source_counts,
)

__all__ = (
    "CoarseRankExampleBatch",
    "FineRankExampleBatch",
    "JoinedSampleAuthorities",
    "NegativeSource",
    "RecallExampleBatch",
    "RequestCandidateTrace",
    "RequestContextBatch",
    "ServingOutput",
    "TraceManifest",
    "corrected_sampled_softmax_loss",
    "negative_source_counts",
)
