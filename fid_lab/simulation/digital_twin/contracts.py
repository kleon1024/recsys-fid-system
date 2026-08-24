"""The only tensor payloads allowed across world/platform boundaries."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch


class Surface(IntEnum):
    FEED = 0
    SEARCH = 1
    COMMERCE = 2
    LIVE = 3
    LOCAL = 4
    POSTING = 5


class ContentKind(IntEnum):
    SHORT_VIDEO = 0
    PHOTO = 1
    ARTICLE = 2
    CARD = 3
    LIVE_ROOM = 4
    PRODUCT = 5
    POI = 6
    AD = 7
    CREATOR_PROMPT = 8


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
    SURFACE_ENTRY = 25
    PLAY_3S = 26
    LONG_VIEW = 27
    COMPLETE = 28
    DWELL = 29
    PUBLISH_FAILED = 30
    MODERATION_REMOVE = 31
    CONTENT_DELETE = 32
    CREATOR_EXIT = 33


class PublishFailureReason(IntEnum):
    NO_CAPACITY = 1
    CREATOR_EXITED = 2
    CREATOR_COOLDOWN = 3


APP_EVENT_SCHEMA_VERSION = 5

EVENT_ORDINAL_BITS = 12
EVENT_TYPE_BITS = 6
EVENT_REQUEST_BITS = 45
EVENT_ORDINAL_MAX = (1 << EVENT_ORDINAL_BITS) - 2
EVENT_TYPE_MAX = (1 << EVENT_TYPE_BITS) - 1
EVENT_REQUEST_MAX = (1 << EVENT_REQUEST_BITS) - 1


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
    query_topic: torch.Tensor

    def __post_init__(self):
        requests = len(self.request_id)
        for name in ("user_id", "surface", "event_time", "query_topic"):
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
    exposure_probability: torch.Tensor
    assignment_probability: torch.Tensor

    def __post_init__(self):
        requests = len(self.request_id)
        if self.item_ids.ndim != 2:
            raise ValueError("rendered item_ids must be [request, position]")
        width = self.item_ids.shape[1]
        _require_shape("item_ids", self.item_ids, (requests, width))
        _require_shape("positions", self.positions, (requests, width))
        _require_shape("valid", self.valid, (requests, width))
        _require_shape(
            "exposure_probability", self.exposure_probability,
            (requests, width),
        )
        for name in (
            "user_id", "surface", "event_time", "ui_variant",
            "assignment_probability",
        ):
            _require_shape(name, getattr(self, name), (requests,))
        if not torch.equal(self.valid, self.item_ids >= 0):
            raise ValueError("slate validity must exactly match nonnegative items")
        if self.valid.any() and (self.positions[self.valid] < 0).any():
            raise ValueError("valid rendered positions must be nonnegative")
        if self.valid.any() and (
            (self.exposure_probability[self.valid] <= 0.0)
            | (self.exposure_probability[self.valid] > 1.0)
        ).any():
            raise ValueError("valid exposure probabilities must be in (0, 1]")
        if requests and (
            (self.assignment_probability <= 0.0)
            | (self.assignment_probability > 1.0)
        ).any():
            raise ValueError("assignment probabilities must be in (0, 1]")

    def select(self, selector) -> RenderedSlateBatch:
        return RenderedSlateBatch(**{
            field.name: getattr(self, field.name)[selector]
            for field in fields(self)
        })

    @classmethod
    def concatenate(cls, batches) -> RenderedSlateBatch:
        batches = tuple(batch for batch in batches if len(batch.request_id))
        if not batches:
            raise ValueError("cannot concatenate an empty slate collection")
        merged = cls(**{
            field.name: torch.cat(tuple(
                getattr(batch, field.name) for batch in batches
            ))
            for field in fields(cls)
        })
        order = torch.argsort(merged.request_id, stable=True)
        ordered = merged.select(order)
        if torch.unique(ordered.request_id).numel() != len(ordered.request_id):
            raise ValueError("rendered requests must be unique")
        return ordered


@dataclass(frozen=True)
class AppEventBatch:
    """Observable, append-only events; no latent state or model output."""

    event_id: torch.Tensor
    schema_version: torch.Tensor
    event_type: torch.Tensor
    event_time: torch.Tensor
    ingest_time: torch.Tensor
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    item_id: torch.Tensor
    post_id: torch.Tensor
    source_candidate_id: torch.Tensor
    creator_id: torch.Tensor
    merchant_id: torch.Tensor
    advertiser_id: torch.Tensor
    product_id: torch.Tensor
    poi_id: torch.Tensor
    order_id: torch.Tensor
    position: torch.Tensor
    content_kind: torch.Tensor
    topic_id: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    query_id: torch.Tensor
    duration_ms: torch.Tensor
    value: torch.Tensor
    logging_probability: torch.Tensor
    assignment_probability: torch.Tensor
    experiment_cell: torch.Tensor

    def __post_init__(self):
        rows = len(self.event_id)
        for field in fields(self):
            _require_shape(field.name, getattr(self, field.name), (rows,))
        if rows and torch.unique(self.event_id).numel() != rows:
            raise ValueError("event batch contains duplicate event_id")
        if (self.ingest_time < self.event_time).any():
            raise ValueError("event ingest time cannot precede event time")
        if rows and not (
            self.schema_version == APP_EVENT_SCHEMA_VERSION
        ).all():
            raise ValueError("event batch contains unsupported schema version")
        for name in ("logging_probability", "assignment_probability"):
            probability = getattr(self, name)
            present = probability >= 0.0
            if present.any() and (
                (probability[present] <= 0.0)
                | (probability[present] > 1.0)
            ).any():
                raise ValueError(f"{name} must be missing or in (0, 1]")
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
            schema_version=torch.full_like(
                integer, APP_EVENT_SCHEMA_VERSION,
            ),
            event_type=integer.clone(),
            event_time=integer.clone(),
            ingest_time=integer.clone(),
            request_id=integer.clone(),
            user_id=integer.clone(),
            surface=integer.clone(),
            item_id=integer.clone(),
            post_id=integer.clone(),
            source_candidate_id=integer.clone(),
            creator_id=integer.clone(),
            merchant_id=integer.clone(),
            advertiser_id=integer.clone(),
            product_id=integer.clone(),
            poi_id=integer.clone(),
            order_id=integer.clone(),
            position=integer.clone(),
            content_kind=integer.clone(),
            topic_id=integer.clone(),
            country=integer.clone(),
            region=integer.clone(),
            query_id=integer.clone(),
            duration_ms=integer.clone(),
            value=torch.empty(0, device=device),
            logging_probability=torch.empty(0, device=device),
            assignment_probability=torch.empty(0, device=device),
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
        order = order[torch.argsort(merged.event_time[order], stable=True)]
        return merged.select(order)

    def event(self, event_type: EventType) -> torch.Tensor:
        return self.event_type == int(event_type)


def deterministic_event_id(
    request_id: torch.Tensor,
    event_type: torch.Tensor,
    item_id: torch.Tensor,
    ordinal: torch.Tensor,
) -> torch.Tensor:
    """Injectively encode the request, event channel and request-local slot.

    Item identity is payload, not event identity. Two items occupying the same
    request/event/ordinal slot are a producer bug and remain visible to the
    duplicate-ID check instead of being hidden behind a probabilistic hash.
    """
    requests = request_id.long()
    event_types = event_type.long()
    ordinals = ordinal.long()
    if ((requests < 0) | (requests > EVENT_REQUEST_MAX)).any():
        raise ValueError("event request_id exceeds the 45-bit identity contract")
    if ((event_types < 0) | (event_types > EVENT_TYPE_MAX)).any():
        raise ValueError("event type exceeds the 6-bit identity contract")
    if ((ordinals < -1) | (ordinals > EVENT_ORDINAL_MAX)).any():
        raise ValueError("event ordinal exceeds the 12-bit identity contract")
    del item_id
    ordinal_code = ordinals + 1
    return (
        requests << (EVENT_TYPE_BITS + EVENT_ORDINAL_BITS)
        | event_types << EVENT_ORDINAL_BITS
        | ordinal_code
    )


def make_app_events(
    event_type: EventType | torch.Tensor,
    *,
    event_time: int | torch.Tensor,
    request_id: torch.Tensor,
    user_id: torch.Tensor,
    surface: torch.Tensor,
    item_id: torch.Tensor | None = None,
    post_id: torch.Tensor | None = None,
    source_candidate_id: torch.Tensor | None = None,
    position: torch.Tensor | None = None,
    experiment_cell: torch.Tensor | None = None,
    content_kind: torch.Tensor | None = None,
    topic_id: torch.Tensor | None = None,
    country: torch.Tensor | None = None,
    region: torch.Tensor | None = None,
    query_id: torch.Tensor | None = None,
    duration_ms: torch.Tensor | None = None,
    creator_id: torch.Tensor | None = None,
    merchant_id: torch.Tensor | None = None,
    advertiser_id: torch.Tensor | None = None,
    product_id: torch.Tensor | None = None,
    poi_id: torch.Tensor | None = None,
    order_id: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    logging_probability: torch.Tensor | None = None,
    assignment_probability: torch.Tensor | None = None,
    ingest_time: int | torch.Tensor | None = None,
    ordinal: torch.Tensor | None = None,
) -> AppEventBatch:
    """Construct one typed, aligned event batch without implicit data access."""
    rows, device = len(request_id), request_id.device
    missing = torch.full((rows,), -1, device=device, dtype=torch.long)
    zeros = torch.zeros(rows, device=device, dtype=torch.long)

    def integer(values):
        return missing.clone() if values is None else values.long()

    def aligned(values):
        if isinstance(values, int):
            return torch.full((rows,), values, device=device, dtype=torch.long)
        return values.long()

    event_types = (
        torch.full((rows,), int(event_type), device=device, dtype=torch.long)
        if isinstance(event_type, EventType) else event_type.long()
    )
    times = aligned(event_time)
    ingest = aligned(event_time if ingest_time is None else ingest_time)
    items = integer(item_id)
    ordinals = zeros if ordinal is None else ordinal.long()
    positions = integer(position)
    return AppEventBatch(
        event_id=deterministic_event_id(
            request_id, event_types, items, ordinals
        ),
        schema_version=torch.full(
            (rows,), APP_EVENT_SCHEMA_VERSION,
            device=device, dtype=torch.long,
        ),
        event_type=event_types,
        event_time=times,
        ingest_time=ingest,
        request_id=request_id.long(),
        user_id=user_id.long(),
        surface=surface.long(),
        item_id=items,
        post_id=integer(post_id),
        source_candidate_id=integer(source_candidate_id),
        creator_id=integer(creator_id),
        merchant_id=integer(merchant_id),
        advertiser_id=integer(advertiser_id),
        product_id=integer(product_id),
        poi_id=integer(poi_id),
        order_id=integer(order_id),
        position=positions,
        content_kind=integer(content_kind),
        topic_id=integer(topic_id),
        country=integer(country),
        region=integer(region),
        query_id=integer(query_id),
        duration_ms=integer(duration_ms),
        value=(
            torch.zeros(rows, device=device)
            if value is None else value.float()
        ),
        logging_probability=(
            torch.full((rows,), -1.0, device=device)
            if logging_probability is None else logging_probability.float()
        ),
        assignment_probability=(
            torch.full((rows,), -1.0, device=device)
            if assignment_probability is None
            else assignment_probability.float()
        ),
        experiment_cell=(
            missing.clone()
            if experiment_cell is None else experiment_cell.long()
        ),
    )
