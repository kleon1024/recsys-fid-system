"""Commerce experiment metrics and preregistered decision gate."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from ...contracts import AppEventBatch, EventType, Surface
from ...experiments.retrieval_ladder import _estimate


def _user_values(events, cell, users, event_type, *, use_value=False):
    selected = (
        events.event(event_type)
        & (events.surface == int(Surface.COMMERCE))
        & (events.experiment_cell == cell)
        & (events.user_id >= 0)
    )
    output = torch.zeros(users, device=events.event_id.device)
    output.index_add_(
        0,
        events.user_id[selected],
        events.value[selected].float()
        if use_value else torch.ones_like(events.value[selected]),
    )
    return output


def _triggered_users(events, cell, users):
    impression = (
        events.event(EventType.IMPRESSION)
        & (events.surface == int(Surface.COMMERCE))
        & (events.experiment_cell == cell)
        & (events.user_id >= 0)
    )
    result = torch.zeros(users, device=events.event_id.device, dtype=torch.bool)
    result[events.user_id[impression]] = True
    return result


def commerce_metrics(events: AppEventBatch, users: int):
    control_users = _triggered_users(events, 1, users)
    treatment_users = _triggered_users(events, 2, users)
    metrics = {}
    for name, event_type, use_value in (
        ("impression", EventType.IMPRESSION, False),
        ("click", EventType.CLICK, False),
        ("detail", EventType.DETAIL, False),
        ("add_cart", EventType.ADD_CART, False),
        ("order", EventType.ORDER, False),
        ("payment", EventType.PAYMENT, False),
        ("refund", EventType.REFUND, False),
        ("paid_value", EventType.PAYMENT, True),
    ):
        control = _user_values(
            events, 1, users, event_type, use_value=use_value,
        )[control_users]
        treatment = _user_values(
            events, 2, users, event_type, use_value=use_value,
        )[treatment_users]
        metrics[name] = _estimate(control, treatment)
    return metrics, {
        "control_triggered_users": int(control_users.sum()),
        "treatment_triggered_users": int(treatment_users.sum()),
    }


def commerce_trace_counts(
    trace, projection, minimum_inventory: float,
) -> dict[str, int]:
    output = {}
    safe = trace.exposed_item_id.clamp_min(0)
    excluded_inventory_product = (
        (trace.exposed_item_id >= 0)
        & (projection.item_inventory[safe] <= minimum_inventory)
        & projection.item_active[safe]
        & (projection.item_product_id[safe] >= 0)
    )
    for cell, name in ((1, "control"), (2, "treatment")):
        rows = (
            (trace.surface == int(Surface.COMMERCE))
            & (trace.experiment_cell == cell)
        )
        output[f"{name}_requests"] = int(rows.sum())
        output[f"{name}_out_of_stock_product_exposures"] = int(
            excluded_inventory_product[rows].sum()
        )
    return output


def merge_counts(
    current: dict[str, int], prior: Mapping[str, object] | None,
) -> dict[str, int]:
    if prior is None:
        return current
    if set(current) != set(prior):
        raise ValueError("Commerce launch counters changed schema")
    return {key: value + int(prior[key]) for key, value in current.items()}


def commerce_decision(metrics, sample, counts, minimum_triggered_users):
    if counts["control_out_of_stock_product_exposures"] == 0:
        return "no_support", "control exposed no inventory-ineligible products"
    if counts["treatment_out_of_stock_product_exposures"]:
        return "reject", "inventory eligibility leaked out-of-stock products"
    if min(sample.values()) < minimum_triggered_users:
        return "hold", "triggered-user sample is below the preregistered gate"
    if not all(
        math.isfinite(value)
        for metric in metrics.values() for value in metric.values()
    ):
        return "hold", "non-finite experiment metric"
    detail = metrics["detail"]
    detail_margin = -0.05 * detail["control_mean"]
    if detail["ci95_high"] < detail_margin:
        return "reject", "detail violates the 5% noninferiority margin"
    if detail["ci95_low"] < detail_margin:
        return "hold", "detail noninferiority is not yet powered"
    cart = metrics["add_cart"]
    if cart["ci95_low"] > 0.0:
        return "promote", "cart improves and detail noninferiority passes"
    if cart["ci95_high"] < 0.0:
        return "reject", "cart significantly decreases"
    return "hold", "cart confidence interval crosses zero"
