"""P3-09a request grouping, cascade and support authority."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    ndcg_score,
    roc_auc_score,
)
import torch

from ....evolution.evaluation.metrics import (
    expected_calibration_error,
    grouped_auc,
)
from ..samples.contracts import FineRankExampleBatch, RequestCandidateTrace


def _count(items: torch.Tensor) -> int:
    return int((items >= 0).sum())


def _trace_tuple(
    trace: RequestCandidateTrace | tuple[RequestCandidateTrace, ...],
) -> tuple[RequestCandidateTrace, ...]:
    return trace if isinstance(trace, tuple) else (trace,)


def stage_report(
    trace: RequestCandidateTrace | tuple[RequestCandidateTrace, ...],
) -> dict[str, object]:
    traces = _trace_tuple(trace)
    counts = {
        name: sum(_count(getattr(value, field)) for value in traces)
        for name, field in {
            "recall": "recall_item_id",
            "coarse": "coarse_item_id",
            "fine": "fine_item_id",
            "exposed": "exposed_item_id",
        }.items()
    }
    request_id = torch.cat(tuple(value.request_id for value in traces))
    requests = len(request_id)
    unique_requests = int(torch.unique(request_id).numel())
    return {
        "requests": requests,
        "unique_requests": unique_requests,
        "duplicate_request_ids": requests - unique_requests,
        "candidate_counts": counts,
        "pass_rate_from_recall": {
            name: value / max(counts["recall"], 1)
            for name, value in counts.items()
        },
        "mean_candidates_per_request": {
            name: value / max(requests, 1) for name, value in counts.items()
        },
    }


def _challenger_probability(
    trace: RequestCandidateTrace,
    challenger_item_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if challenger_item_id.ndim != 2 or (
        challenger_item_id.shape[0] != len(trace.request_id)
    ):
        raise ValueError("challenger slate must be request aligned")
    match = (
        challenger_item_id[:, :, None]
        == trace.recall_item_id[:, None, :]
    )
    location = match.float().argmax(dim=2)
    probability = torch.gather(
        trace.candidate_exposure_probability, 1, location,
    )
    present = match.any(dim=2) & (challenger_item_id >= 0)
    return torch.where(present, probability, torch.zeros_like(probability)), present


def support_report(
    trace: RequestCandidateTrace | tuple[RequestCandidateTrace, ...],
    challenger_item_id: torch.Tensor | None = None,
) -> dict[str, object]:
    traces = _trace_tuple(trace)
    if len(traces) > 1:
        if challenger_item_id is not None:
            raise ValueError("multi-partition challenger support must stay partitioned")
        reports = tuple(support_report(value) for value in traces)
        candidates = sum(int(value["candidate_rows"]) for value in reports)
        supported = sum(
            int(value["randomized_supported_candidate_rows"])
            for value in reports
        )
        factual = sum(int(value["factual_exposures"]) for value in reports)
        factual_supported = sum(
            int(value["factual_exposures_with_propensity"])
            for value in reports
        )
        return {
            "randomized_requests": sum(
                int(value["randomized_requests"]) for value in reports
            ),
            "candidate_rows": candidates,
            "randomized_supported_candidate_rows": supported,
            "randomized_supported_candidate_rate": supported / max(candidates, 1),
            "factual_exposures": factual,
            "factual_exposures_with_propensity": factual_supported,
            "factual_action_support_complete": factual == factual_supported,
            "candidate_ope_identified": False,
            "slate_ope_identified": False,
        }
    trace = traces[0]
    valid = trace.recall_item_id >= 0
    randomized_lane = trace.exploration_rate > 0.0
    supported = (
        valid
        & randomized_lane[:, None]
        & (trace.candidate_exposure_probability > 0.0)
    )
    exposed = trace.exposed_item_id >= 0
    factual_supported = exposed & (trace.exposure_probability > 0.0)
    report: dict[str, object] = {
        "randomized_requests": int(randomized_lane.sum()),
        "candidate_rows": int(valid.sum()),
        "randomized_supported_candidate_rows": int(supported.sum()),
        "randomized_supported_candidate_rate": float(
            supported.sum() / valid.sum().clamp_min(1)
        ),
        "factual_exposures": int(exposed.sum()),
        "factual_exposures_with_propensity": int(factual_supported.sum()),
        "factual_action_support_complete": bool(
            torch.equal(exposed, factual_supported)
        ),
        "candidate_ope_identified": False,
        "slate_ope_identified": False,
    }
    if challenger_item_id is None:
        return report
    probability, present = _challenger_probability(trace, challenger_item_id)
    valid_challenger = challenger_item_id >= 0
    challenger_supported = (
        valid_challenger
        & present
        & randomized_lane[:, None]
        & (probability > 0.0)
    )
    exact_slate = (
        challenger_item_id.shape == trace.exposed_item_id.shape
        and torch.equal(challenger_item_id, trace.exposed_item_id)
    )
    report.update({
        "challenger_rows": int(valid_challenger.sum()),
        "challenger_rows_in_logged_corpus": int(
            (valid_challenger & present).sum()
        ),
        "challenger_supported_rows": int(challenger_supported.sum()),
        "candidate_ope_identified": bool(
            torch.equal(valid_challenger, challenger_supported)
        ),
        "slate_replay_exact": exact_slate,
        "slate_ope_identified": exact_slate,
    })
    return report


def _safe_binary_metrics(
    label: np.ndarray,
    probability: np.ndarray,
    request_id: np.ndarray,
) -> dict[str, object]:
    prevalence = float(label.mean()) if len(label) else 0.0
    if len(np.unique(label)) < 2:
        return {
            "rows": len(label),
            "prevalence": prevalence,
            "auc": None,
            "pr_auc": None,
            "log_loss": None,
            "normalized_entropy": None,
            "ece": None,
            "request_gauc": grouped_auc(label, probability, request_id),
        }
    clipped = np.clip(probability.astype(float), 1e-7, 1.0 - 1e-7)
    entropy = -(
        prevalence * np.log(max(prevalence, 1e-7))
        + (1.0 - prevalence) * np.log(max(1.0 - prevalence, 1e-7))
    )
    loss = float(log_loss(label, clipped))
    return {
        "rows": len(label),
        "prevalence": prevalence,
        "auc": float(roc_auc_score(label, clipped)),
        "pr_auc": float(average_precision_score(label, clipped)),
        "log_loss": loss,
        "normalized_entropy": loss / max(entropy, 1e-7),
        "ece": expected_calibration_error(label, clipped),
        "request_gauc": grouped_auc(label, clipped, request_id),
    }


def _request_ndcg(
    label: np.ndarray,
    score: np.ndarray,
    request_id: np.ndarray,
) -> dict[str, object]:
    values = []
    for group in np.unique(request_id):
        selected = request_id == group
        if selected.sum() < 2 or label[selected].sum() <= 0:
            continue
        values.append(float(ndcg_score(
            label[selected][None, :], score[selected][None, :],
        )))
    return {
        "value": None if not values else float(np.mean(values)),
        "eligible_requests": len(values),
        "total_requests": int(np.unique(request_id).size),
    }


def evaluate_request_batch(
    trace: RequestCandidateTrace,
    fine: FineRankExampleBatch,
    *,
    rank_scores: torch.Tensor | None = None,
    probabilities: torch.Tensor | None = None,
    challenger_item_id: torch.Tensor | None = None,
) -> dict[str, object]:
    if not torch.equal(trace.request_id, fine.request_id):
        raise ValueError("evaluation trace and fine examples are misaligned")
    if rank_scores is not None and rank_scores.shape != fine.labels.shape:
        raise ValueError("rank scores must align with fine labels")
    if probabilities is not None and probabilities.shape != fine.labels.shape:
        raise ValueError("probabilities must align with fine labels")
    task_metrics = {}
    request = fine.request_id[:, None].expand_as(fine.item_id)
    for index, name in enumerate(fine.task_names):
        mask = fine.label_mask[:, :, index]
        if not mask.any():
            task_metrics[name] = {"rows": 0, "status": "no_mature_labels"}
            continue
        label = fine.labels[:, :, index][mask].detach().cpu().numpy()
        group = request[mask].detach().cpu().numpy()
        metric: dict[str, object] = {
            "rows": len(label),
            "prevalence": float(label.mean()),
        }
        if probabilities is not None:
            probability = probabilities[:, :, index][mask].detach().cpu().numpy()
            metric.update(_safe_binary_metrics(label, probability, group))
        if rank_scores is not None:
            score = rank_scores[:, :, index][mask].detach().cpu().numpy()
            metric["request_ndcg"] = _request_ndcg(label, score, group)
        task_metrics[name] = metric
    return {
        "schema": "p3-request-aware-evaluation/v1",
        "grouping": "request_id",
        "stage": stage_report(trace),
        "support": support_report(trace, challenger_item_id),
        "tasks": task_metrics,
    }
