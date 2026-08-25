"""Ads/Feed experiment metrics and causal launch gate."""

from __future__ import annotations

import math

import torch

from ...contracts import AppEventBatch, ContentKind, EventType, Surface
from ...experiments.retrieval_ladder import _estimate
from .audit import AdsMarketAudit


def _triggered(events, cell, users):
    impression = (
        events.event(EventType.IMPRESSION)
        & (events.surface == int(Surface.FEED))
        & (events.experiment_cell == cell)
        & (events.user_id >= 0)
    )
    result = torch.zeros(users, device=events.event_id.device, dtype=torch.bool)
    result[events.user_id[impression]] = True
    return result


def _user_value(events, cell, users, event_type, *, ads=False, value=False):
    selected = (
        events.event(event_type)
        & (events.surface == int(Surface.FEED))
        & (events.experiment_cell == cell)
        & (events.user_id >= 0)
    )
    if ads:
        selected &= events.content_kind == int(ContentKind.AD)
    output = torch.zeros(users, device=events.event_id.device)
    if event_type == EventType.DWELL:
        increment = events.duration_ms[selected].float() / 1_000.0
    elif value:
        increment = events.value[selected].float()
    else:
        increment = torch.ones_like(events.value[selected])
    output.index_add_(0, events.user_id[selected], increment)
    return output


def ads_metrics(events: AppEventBatch, users: int):
    cohort = {
        cell: _triggered(events, cell, users) for cell in (0, 1)
    }
    definitions = (
        ("dwell_seconds", EventType.DWELL, False, False),
        ("play_3s", EventType.PLAY_3S, False, False),
        ("long_view", EventType.LONG_VIEW, False, False),
        ("negative", EventType.NEGATIVE, False, False),
        ("session_end", EventType.SESSION_END, False, False),
        ("ad_impression", EventType.IMPRESSION, True, False),
        ("ad_click", EventType.CLICK, True, False),
        ("ad_spend", EventType.AD_SPEND, True, True),
        ("pixel_conversion", EventType.PIXEL_CONVERSION, True, False),
        ("pixel_value", EventType.PIXEL_CONVERSION, True, True),
    )
    metrics = {}
    for name, event_type, ads, value in definitions:
        control = _user_value(
            events, 0, users, event_type, ads=ads, value=value,
        )[cohort[0]]
        treatment = _user_value(
            events, 1, users, event_type, ads=ads, value=value,
        )[cohort[1]]
        metrics[name] = _estimate(control, treatment)
    return metrics, {
        "control_triggered_users": int(cohort[0].sum()),
        "treatment_triggered_users": int(cohort[1].sum()),
    }


def ads_trace_counts(trace, content_kind) -> dict[str, int]:
    route = trace.manifest.route_names.index("ads_auction")
    exposed_ad = content_kind[trace.exposed_item_id.clamp_min(0)] == int(
        ContentKind.AD
    )
    output = {}
    for cell, name in ((0, "control"), (1, "treatment")):
        rows = (
            (trace.surface == int(Surface.FEED))
            & (trace.experiment_cell == cell)
        )
        output[f"{name}_requests"] = int(rows.sum())
        output[f"{name}_auction_candidates"] = int(
            trace.route_valid[rows, route].sum()
        )
        output[f"{name}_ad_exposures"] = int(exposed_ad[rows].sum())
    return output


def merge_trace_counts(current, prior):
    if prior is None:
        return current
    if set(current) != set(prior):
        raise ValueError("Ads launch counters changed schema")
    return {key: value + int(prior[key]) for key, value in current.items()}


def ads_decision(metrics, sample, audit: AdsMarketAudit, minimum_users):
    if metrics["ad_impression"]["control_mean"] > 0.0:
        return "reject", "Ads auction leaked into control traffic"
    if (
        metrics["ad_impression"]["treatment_mean"] <= 0.0
        or audit.impressions == 0
        or audit.billed_revenue <= 0.0
    ):
        return "no_support", "treatment produced no billable Ads support"
    if any((
        audit.unbudgeted_spend,
        audit.unpriced_spend,
        audit.over_bid_spend,
        audit.overspend_events,
        audit.partially_billed_impressions,
    )):
        return "reject", "Ads budget or billing reconciliation failed"
    if audit.maximum_ads_per_request > 1:
        return "reject", "Ads load exceeds the one-per-slate constraint"
    if min(sample.values()) < minimum_users:
        return "hold", "triggered-user sample is below the preregistered gate"
    if not all(
        math.isfinite(value)
        for metric in metrics.values() for value in metric.values()
    ):
        return "hold", "non-finite experiment metric"
    dwell = metrics["dwell_seconds"]
    dwell_margin = -0.01 * dwell["control_mean"]
    if dwell["ci95_high"] < dwell_margin:
        return "reject", "Feed stay violates the 1% noninferiority margin"
    if metrics["negative"]["ci95_low"] > 0.0:
        return "reject", "negative feedback significantly increases"
    if dwell["ci95_low"] < dwell_margin:
        return "hold", "Feed-stay noninferiority is not yet powered"
    return "promote", "billable revenue improves with Feed guardrails passing"
