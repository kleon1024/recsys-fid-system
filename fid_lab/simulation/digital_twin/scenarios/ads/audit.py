"""Validate Ads exposure, billing, budget and delayed conversion lineage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...contracts import AppEventBatch, ContentKind, EventType


@dataclass(frozen=True)
class AdsMarketAudit:
    impressions: int
    clicks: int
    spend_events: int
    pixel_conversions: int
    billed_revenue: float
    unbudgeted_spend: int
    unpriced_spend: int
    over_bid_spend: int
    overspend_events: int
    partially_billed_impressions: int
    maximum_ads_per_request: int


def _decision_keys(events: AppEventBatch, mask) -> list[tuple[int, int, int]]:
    return list(zip(
        events.request_id[mask].tolist(),
        events.item_id[mask].tolist(),
        events.position[mask].tolist(),
        strict=True,
    ))


def _validate_event_lineage(
    events: AppEventBatch, scope,
) -> tuple[object, object]:
    impression = (
        events.event(EventType.IMPRESSION)
        & (events.content_kind == int(ContentKind.AD))
        & scope
    )
    spend = events.event(EventType.AD_SPEND) & scope
    impression_keys = _decision_keys(events, impression)
    spend_keys = _decision_keys(events, spend)
    if len(spend_keys) != len(set(spend_keys)):
        raise ValueError("Ads billing contains duplicate decision keys")
    if set(impression_keys) != set(spend_keys):
        raise ValueError("every ad impression must have exactly one spend event")
    click = events.event(EventType.CLICK) & (
        events.content_kind == int(ContentKind.AD)
    )
    pixel = events.event(EventType.PIXEL_CONVERSION) & scope
    if not set(_decision_keys(events, pixel)).issubset(
        set(_decision_keys(events, click))
    ):
        raise ValueError("Pixel conversion has no attributed ad click")
    return impression, spend


def _market_violations(
    events: AppEventBatch, scope,
) -> tuple[int, int, int, int, int]:
    selected = (
        events.event(EventType.AD_BUDGET)
        | events.event(EventType.BID)
        | (events.event(EventType.AD_SPEND) & scope)
    ) & (events.advertiser_id >= 0)
    selected_type = events.event_type[selected]
    selected_ingest = events.ingest_time[selected]
    row = selected_type.argsort(stable=True)
    row = row[selected_ingest[row].argsort(stable=True)]
    index = selected.nonzero().flatten()[row]
    budget: dict[int, float] = {}
    bid: dict[int, float] = {}
    unbudgeted = unpriced = over_bid = overspend = partial = 0
    for position in index.tolist():
        advertiser = int(events.advertiser_id[position])
        value = float(events.value[position])
        event_type = int(events.event_type[position])
        if event_type == int(EventType.AD_BUDGET):
            budget[advertiser] = max(value, 0.0)
            continue
        if event_type == int(EventType.BID):
            bid[advertiser] = max(value, 0.0)
            continue
        current = _spend_violations(advertiser, value, budget, bid)
        unbudgeted += current[0]
        unpriced += current[1]
        over_bid += current[2]
        overspend += current[3]
        partial += current[4]
    return unbudgeted, unpriced, over_bid, overspend, partial


def _spend_violations(advertiser, value, budget, bid):
    unbudgeted = int(advertiser not in budget)
    unpriced = int(advertiser not in bid or bid.get(advertiser, 0.0) <= 0.0)
    price = bid.get(advertiser, 0.0)
    over_bid = int(not unpriced and value > price + 1e-6)
    partial = int(not unpriced and value + 1e-6 < price)
    remaining = budget.get(advertiser, 0.0)
    overspend = int(value > remaining + 1e-6)
    budget[advertiser] = max(remaining - value, 0.0)
    return unbudgeted, unpriced, over_bid, overspend, partial


def audit_ads_market(
    events: AppEventBatch, *, start_time: int | None = None,
) -> AdsMarketAudit:
    scope = (
        events.ingest_time >= start_time
        if start_time is not None
        else events.ingest_time >= torch.iinfo(events.ingest_time.dtype).min
    )
    impression, spend = _validate_event_lineage(events, scope)
    violations = _market_violations(events, scope)
    click = events.event(EventType.CLICK) & (
        events.content_kind == int(ContentKind.AD)
    )
    click &= scope
    pixel = events.event(EventType.PIXEL_CONVERSION) & scope
    request_counts: dict[int, int] = {}
    for request in events.request_id[impression].tolist():
        request_counts[request] = request_counts.get(request, 0) + 1
    return AdsMarketAudit(
        impressions=int(impression.sum()),
        clicks=int(click.sum()),
        spend_events=int(spend.sum()),
        pixel_conversions=int(pixel.sum()),
        billed_revenue=float(events.value[spend].sum()),
        unbudgeted_spend=violations[0],
        unpriced_spend=violations[1],
        over_bid_spend=violations[2],
        overspend_events=violations[3],
        partially_billed_impressions=violations[4],
        maximum_ads_per_request=max(request_counts.values(), default=0),
    )
