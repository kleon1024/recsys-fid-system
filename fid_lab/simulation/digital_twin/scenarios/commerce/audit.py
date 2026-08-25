"""Fail-closed lineage checks for the Commerce transaction funnel."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...contracts import AppEventBatch, ContentKind, EventType, Surface


@dataclass(frozen=True)
class CommerceFunnelAudit:
    impressions: int
    clicks: int
    details: int
    carts: int
    orders: int
    payments: int
    refunds: int
    paid_value: float
    refunded_value: float

    @property
    def net_paid_value(self) -> float:
        return self.paid_value - self.refunded_value


def _commerce(events: AppEventBatch, event_type: EventType) -> torch.Tensor:
    return events.event(event_type) & (
        events.surface == int(Surface.COMMERCE)
    )


def _request_position_key(events: AppEventBatch, selected: torch.Tensor):
    return events.request_id[selected] * 10_000 + events.position[selected]


def audit_commerce_funnel(events: AppEventBatch) -> CommerceFunnelAudit:
    """Validate conditional spaces before reporting Commerce conversion."""
    impression = _commerce(events, EventType.IMPRESSION)
    click = _commerce(events, EventType.CLICK)
    detail = _commerce(events, EventType.DETAIL)
    cart = _commerce(events, EventType.ADD_CART)
    order = _commerce(events, EventType.ORDER)
    payment = _commerce(events, EventType.PAYMENT)
    refund = _commerce(events, EventType.REFUND)
    if not torch.isin(
        _request_position_key(events, cart),
        _request_position_key(events, detail),
    ).all():
        raise ValueError("Commerce cart exists outside the detail space")
    if cart.any() and (
        events.content_kind[cart] != int(ContentKind.PRODUCT)
    ).any():
        raise ValueError("Commerce cart must reference a product")
    if not torch.isin(
        events.order_id[order],
        events.order_id[cart],
    ).all():
        raise ValueError("Commerce order exists outside the cart space")
    if not torch.isin(
        events.order_id[payment],
        events.order_id[order],
    ).all():
        raise ValueError("Commerce payment exists outside the order space")
    if not torch.isin(
        events.order_id[refund],
        events.order_id[payment],
    ).all():
        raise ValueError("Commerce refund exists outside the payment space")
    return CommerceFunnelAudit(
        impressions=int(impression.sum()),
        clicks=int(click.sum()),
        details=int(detail.sum()),
        carts=int(cart.sum()),
        orders=int(order.sum()),
        payments=int(payment.sum()),
        refunds=int(refund.sum()),
        paid_value=float(events.value[payment].sum()),
        refunded_value=float(-events.value[refund].clamp_max(0.0).sum()),
    )
