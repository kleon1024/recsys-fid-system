"""Search experiment metrics and preregistered decision gate."""

from __future__ import annotations

import math

import torch

from ...contracts import AppEventBatch, EventType, Surface
from ...experiments.retrieval_ladder import _estimate


def _query_cohort(
    events: AppEventBatch, cell: int, users: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assigned_impression = (
        events.event(EventType.IMPRESSION)
        & (events.surface == int(Surface.SEARCH))
        & (events.experiment_cell == cell)
    )
    assigned_request = torch.unique(
        events.request_id[assigned_impression], sorted=True,
    )
    query = (
        events.event(EventType.QUERY)
        & (events.surface == int(Surface.SEARCH))
        & (events.user_id >= 0)
        & torch.isin(events.request_id, assigned_request)
    )
    query_count = torch.zeros(users, device=events.event_id.device)
    query_count.index_add_(
        0, events.user_id[query], torch.ones_like(events.value[query]),
    )
    return query, query_count, query_count > 0


def _request_rate_by_user(
    events: AppEventBatch,
    query: torch.Tensor,
    query_count: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    query_request = events.request_id[query]
    query_user = events.user_id[query]
    target_request = torch.unique(events.request_id[target], sorted=True)
    matched = torch.isin(query_request, target_request).float()
    count = torch.zeros_like(query_count)
    count.index_add_(0, query_user, matched)
    return count / query_count.clamp_min(1.0)


def _post_search_feed_rate(
    events: AppEventBatch,
    query: torch.Tensor,
    query_count: torch.Tensor,
) -> torch.Tensor:
    post_feed = (
        events.event(EventType.SURFACE_ENTRY)
        & (events.surface == int(Surface.FEED))
        & (events.query_id >= 0)
    )
    followed_query = torch.unique(events.query_id[post_feed], sorted=True)
    matched = torch.isin(events.query_id[query], followed_query).float()
    result = torch.zeros_like(query_count)
    result.index_add_(0, events.user_id[query], matched)
    return result / query_count.clamp_min(1.0)


def search_metrics(
    events: AppEventBatch, users: int,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Estimate request-normalized outcomes with user-clustered observations."""
    metrics: dict[str, dict[str, float]] = {}
    samples: dict[str, int] = {}
    values: dict[int, dict[str, torch.Tensor]] = {}
    for cell, name in ((0, "control"), (1, "treatment")):
        query, query_count, triggered = _query_cohort(events, cell, users)
        samples[f"{name}_triggered_users"] = int(triggered.sum())
        scoped = events.experiment_cell == cell
        values[cell] = {
            "success_rate": _request_rate_by_user(
                events,
                query,
                query_count,
                events.event(EventType.SEARCH_SUCCESS) & scoped,
            )[triggered],
            "reformulation_rate": _request_rate_by_user(
                events,
                query,
                query_count,
                events.event(EventType.SEARCH_REFORMULATE) & scoped,
            )[triggered],
            "abandonment_rate": _request_rate_by_user(
                events,
                query,
                query_count,
                events.event(EventType.SEARCH_ABANDON) & scoped,
            )[triggered],
            "detail_rate": _request_rate_by_user(
                events,
                query,
                query_count,
                events.event(EventType.DETAIL) & scoped,
            )[triggered],
            "post_search_feed_rate": _post_search_feed_rate(
                events, query, query_count,
            )[triggered],
        }
    for metric in values[0]:
        metrics[metric] = _estimate(values[0][metric], values[1][metric])
    return metrics, samples


def semantic_route_counts(trace) -> dict[str, int]:
    route = trace.manifest.route_names.index("search_semantic")
    output: dict[str, int] = {}
    for cell, name in ((0, "control"), (1, "treatment")):
        rows = (
            (trace.surface == int(Surface.SEARCH))
            & (trace.experiment_cell == cell)
        )
        output[f"{name}_requests"] = int(rows.sum())
        output[f"{name}_semantic_candidates"] = int(
            trace.route_valid[rows, route].sum()
        )
    return output


def merge_route_counts(
    current: dict[str, int], prior: dict[str, int] | None,
) -> dict[str, int]:
    if prior is None:
        return current
    if set(current) != set(prior):
        raise ValueError("Search launch counters changed schema")
    return {key: value + int(prior[key]) for key, value in current.items()}


def search_decision(metrics, sample, counts, minimum_triggered_users):
    if counts["treatment_semantic_candidates"] == 0:
        return "reject", "semantic retrieval supplied no treatment candidates"
    if counts["control_semantic_candidates"]:
        return "reject", "semantic retrieval leaked into control"
    if min(sample.values()) < minimum_triggered_users:
        return "hold", "triggered-user sample is below the preregistered gate"
    if not all(
        math.isfinite(value)
        for metric in metrics.values() for value in metric.values()
    ):
        return "hold", "non-finite experiment metric"
    success = metrics["success_rate"]
    reformulation = metrics["reformulation_rate"]
    detail = metrics["detail_rate"]
    detail_margin = -0.03 * detail["control_mean"]
    if detail["ci95_high"] < detail_margin:
        return "reject", "detail rate violates the 3% noninferiority margin"
    if reformulation["ci95_low"] > 0.0:
        return "reject", "query reformulation significantly increases"
    if success["ci95_high"] < 0.0:
        return "reject", "Search success significantly decreases"
    if detail["ci95_low"] < detail_margin:
        return "hold", "detail noninferiority is not yet powered"
    if success["ci95_low"] <= 0.0:
        return "hold", "Search-success confidence interval crosses zero"
    return "promote", "Search success improves with guardrails passing"
