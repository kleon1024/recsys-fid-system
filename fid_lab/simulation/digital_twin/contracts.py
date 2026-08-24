"""The only tensor payloads allowed across world/platform boundaries."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch


class EventType(IntEnum):
    REGISTRATION = 0
    SESSION_START = 1
    QUERY = 2
    IMPRESSION = 3
    EXAMINE = 4
    PLAY = 5
    SLIDE = 6
    CLICK = 7
    LIKE = 8
    COMMENT = 9
    SHARE = 10
    FOLLOW = 11
    NEGATIVE = 12
    DETAIL = 13
    FAVORITE = 14
    ADD_CART = 15
    ORDER = 16
    PAYMENT = 17
    REFUND = 18
    PIXEL_CONVERSION = 19
    CREATE = 20
    PUBLISH = 21
    SESSION_END = 22
    INVENTORY = 23
    BID = 24


def _require_shape(name, value, shape):
    if value.shape != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")


@dataclass(frozen=True)
class PlatformRequestBatch:
    """Observable requests opened from entry/query/navigation events."""

    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    event_time: torch.Tensor

    def __post_init__(self):
        requests = len(self.request_id)
        for name in ("user_id", "surface", "event_time"):
            _require_shape(name, getattr(self, name), (requests,))

    def select(self, selector) -> PlatformRequestBatch:
        return PlatformRequestBatch(**{
            field.name: getattr(self, field.name)[selector]
            for field in fields(self)
        })


@dataclass(frozen=True)
class RenderedSlateBatch:
    """What the app rendered; deliberately excludes scores and features."""

    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    event_time: torch.Tensor
    item_ids: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    ui_variant: torch.Tensor

    def __post_init__(self):
        requests = len(self.request_id)
        if self.item_ids.ndim != 2:
            raise ValueError("rendered item_ids must be [request, position]")
        width = self.item_ids.shape[1]
        _require_shape("item_ids", self.item_ids, (requests, width))
        _require_shape("positions", self.positions, (requests, width))
        _require_shape("valid", self.valid, (requests, width))
        for name in (
            "user_id", "surface", "event_time", "ui_variant",
        ):
            _require_shape(name, getattr(self, name), (requests,))
        if not torch.equal(self.valid, self.item_ids >= 0):
            raise ValueError("slate validity must exactly match nonnegative items")
        if self.valid.any() and (self.positions[self.valid] < 0).any():
            raise ValueError("valid rendered positions must be nonnegative")

    def select(self, selector) -> RenderedSlateBatch:
        return RenderedSlateBatch(**{
            field.name: getattr(self, field.name)[selector]
            for field in fields(self)
        })


@dataclass(frozen=True)
class AppEventBatch:
    """Observable, append-only events; no latent state or model output."""

    event_id: torch.Tensor
    event_type: torch.Tensor
    event_time: torch.Tensor
    ingest_time: torch.Tensor
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    item_id: torch.Tensor
    creator_id: torch.Tensor
    product_id: torch.Tensor
    poi_id: torch.Tensor
    order_id: torch.Tensor
    position: torch.Tensor
    value: torch.Tensor
    experiment_cell: torch.Tensor

    def __post_init__(self):
        rows = len(self.event_id)
        for field in fields(self):
            _require_shape(field.name, getattr(self, field.name), (rows,))
        if rows and torch.unique(self.event_id).numel() != rows:
            raise ValueError("event batch contains duplicate event_id")
        if (self.ingest_time < self.event_time).any():
            raise ValueError("event ingest time cannot precede event time")
        if rows and (
            (self.event_type < min(EventType))
            | (self.event_type > max(EventType))
        ).any():
            raise ValueError("event batch contains unknown event type")

    @classmethod
    def empty(cls, device="cpu") -> AppEventBatch:
        integer = torch.empty(0, device=device, dtype=torch.long)
        return cls(
            event_id=integer,
            event_type=integer.clone(),
            event_time=integer.clone(),
            ingest_time=integer.clone(),
            request_id=integer.clone(),
            user_id=integer.clone(),
            surface=integer.clone(),
            item_id=integer.clone(),
            creator_id=integer.clone(),
            product_id=integer.clone(),
            poi_id=integer.clone(),
            order_id=integer.clone(),
            position=integer.clone(),
            value=torch.empty(0, device=device),
            experiment_cell=integer.clone(),
        )

    def select(self, selector) -> AppEventBatch:
        return AppEventBatch(**{
            field.name: getattr(self, field.name)[selector]
            for field in fields(self)
        })

    @classmethod
    def concatenate(cls, batches) -> AppEventBatch:
        batches = tuple(batch for batch in batches if len(batch.event_id))
        if not batches:
            return cls.empty()
        merged = cls(**{
            field.name: torch.cat(tuple(
                getattr(batch, field.name) for batch in batches
            ))
            for field in fields(cls)
        })
        order = torch.argsort(merged.event_id, stable=True)
        return merged.select(order)

    def event(self, event_type: EventType) -> torch.Tensor:
        return self.event_type == int(event_type)


def deterministic_event_id(
    request_id: torch.Tensor,
    event_type: torch.Tensor,
    item_id: torch.Tensor,
    ordinal: torch.Tensor,
) -> torch.Tensor:
    """Stable event identity independent of batch and experiment-cell order."""
    mask = 0x7FFFFFFFFFFFFFFF
    value = request_id.long() * 1_103_515_245
    value += event_type.long() * 104_729
    value += item_id.long().clamp_min(0) * 48_271
    value += ordinal.long() * 12_345
    value = torch.bitwise_and(value, mask)
    value = torch.bitwise_xor(value, torch.bitwise_right_shift(value, 29))
    return torch.bitwise_and(value * 2_654_435_761, mask)
