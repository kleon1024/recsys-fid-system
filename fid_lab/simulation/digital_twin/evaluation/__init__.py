"""Request-aware offline, support and factual experiment evaluation."""

from .experiment import FactualABAccumulator, aa_decision, factual_ab_report
from .request import (
    RequestWindowAccumulator,
    evaluate_request_batch,
    stage_report,
    support_report,
)

__all__ = (
    "aa_decision",
    "FactualABAccumulator",
    "RequestWindowAccumulator",
    "evaluate_request_batch",
    "factual_ab_report",
    "stage_report",
    "support_report",
)
