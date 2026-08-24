"""Watermark-aware materializer for recall, coarse and fine authorities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...randomness.counter import uniform_for_items
from ..catalog import PublicCatalog
from ..contracts import AppEventBatch, ContentKind, EventType, Surface
from ..contracts import deterministic_event_id
from ..platform.projection import ProjectionSnapshot
from .contracts import (
    CoarseRankExampleBatch,
    FineRankExampleBatch,
    JoinedSampleAuthorities,
    RecallExampleBatch,
    RequestCandidateTrace,
    RequestContextBatch,
)


@dataclass(frozen=True)
class LabelTask:
    name: str
    event_type: EventType
    maturity_ticks: int
    weight: float


@dataclass(frozen=True)
class JoinerConfig:
    ticks_per_day: int
    recall_negatives: int = 20
    sampling_seed: int = 1_607

    def __post_init__(self):
        if self.ticks_per_day <= 0 or self.recall_negatives <= 0:
            raise ValueError("joiner dimensions must be positive")

    @property
    def tasks(self) -> tuple[LabelTask, ...]:
        day = self.ticks_per_day
        return (
            LabelTask("play", EventType.PLAY, 0, 0.05),
            LabelTask("play_3s", EventType.PLAY_3S, 0, 0.12),
            LabelTask("long_view", EventType.LONG_VIEW, 0, 0.45),
            LabelTask("complete", EventType.COMPLETE, 0, 0.65),
            LabelTask("click", EventType.CLICK, 0, 0.20),
            LabelTask("like", EventType.LIKE, 0, 0.35),
            LabelTask("share", EventType.SHARE, 0, 0.60),
            LabelTask("negative", EventType.NEGATIVE, 0, -0.80),
            LabelTask("detail", EventType.DETAIL, 0, 0.30),
            LabelTask("favorite", EventType.FAVORITE, 0, 0.50),
            LabelTask("add_cart", EventType.ADD_CART, 0, 0.70),
            LabelTask("order", EventType.ORDER, 12, 1.00),
            LabelTask("payment", EventType.PAYMENT, 16, 1.30),
            LabelTask("refund", EventType.REFUND, 15 * day, -1.10),
            LabelTask(
                "pixel_conversion",
                EventType.PIXEL_CONVERSION,
                3 * day,
                1.00,
            ),
            LabelTask("create", EventType.CREATE, 0, 0.55),
            LabelTask("publish", EventType.PUBLISH, 0, 1.00),
        )


def capture_request_context(
    trace: RequestCandidateTrace,
    projection: ProjectionSnapshot,
) -> RequestContextBatch:
    if projection.as_of_ingest_time > int(trace.event_time.min()):
        raise ValueError("projection snapshot is later than a request")
    user = trace.user_id
    state = projection.state
    as_of = torch.full_like(trace.event_time, projection.as_of_ingest_time)
    return RequestContextBatch(
        request_id=trace.request_id,
        request_time=trace.event_time,
        user_event_counts=state.user_event_counts[user].clone(),
        user_surface_counts=state.user_surface_counts[user].clone(),
        history_item_id=state.user_history_item[user].clone(),
        history_event_time=state.user_history_event_time[user].clone(),
        history_ingest_time=state.user_history_ingest_time[user].clone(),
        feature_as_of_ingest_time=as_of,
    )


def _event_values(
    events: AppEventBatch,
    event_type: EventType,
    trace: RequestCandidateTrace,
    *,
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = events.event(event_type)
    event_id = events.event_id[selected]
    output = torch.zeros_like(trace.exposed_item_id, dtype=torch.float)
    if not len(event_id):
        return output
    event_value = (
        torch.ones_like(event_id, dtype=torch.float)
        if values is None else values[selected].float()
    )
    order = torch.argsort(event_id)
    event_id, event_value = event_id[order], event_value[order]
    requests = trace.request_id[:, None].expand_as(trace.exposed_item_id)
    event_types = torch.full_like(requests, int(event_type))
    query_id = deterministic_event_id(
        requests,
        event_types,
        trace.exposed_item_id,
        trace.exposed_position,
    )
    location = torch.searchsorted(event_id, query_id).clamp_max(len(event_id) - 1)
    matched = event_id[location] == query_id
    output[matched] = event_value[location[matched]]
    return output


def _task_applicability(
    task: LabelTask,
    surface: torch.Tensor,
    content_kind: torch.Tensor,
) -> torch.Tensor:
    feed_or_live = (
        (surface == int(Surface.FEED))
        | (surface == int(Surface.LIVE))
    )
    decision_surface = (
        (surface == int(Surface.SEARCH))
        | (surface == int(Surface.COMMERCE))
        | (surface == int(Surface.LOCAL))
        | (surface == int(Surface.POSTING))
    )
    if task.event_type in {
        EventType.PLAY,
        EventType.PLAY_3S,
        EventType.LONG_VIEW,
        EventType.COMPLETE,
        EventType.LIKE,
        EventType.SHARE,
    }:
        return feed_or_live
    if task.event_type in {EventType.CLICK, EventType.DETAIL}:
        return decision_surface
    if task.event_type == EventType.FAVORITE:
        return (
            (surface == int(Surface.SEARCH))
            | (surface == int(Surface.LOCAL))
        )
    if task.event_type == EventType.ADD_CART:
        return surface == int(Surface.COMMERCE)
    if task.event_type in {
        EventType.ORDER,
        EventType.PAYMENT,
        EventType.REFUND,
    }:
        return (
            (surface == int(Surface.COMMERCE))
            | (surface == int(Surface.LOCAL))
        )
    if task.event_type == EventType.PIXEL_CONVERSION:
        return content_kind == int(ContentKind.AD)
    if task.event_type in {EventType.CREATE, EventType.PUBLISH}:
        return surface == int(Surface.POSTING)
    return torch.ones_like(surface, dtype=torch.bool)


def _map_stage(
    child: torch.Tensor,
    parent: torch.Tensor,
    parent_value: torch.Tensor,
    missing_value: float | int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sentinel = torch.iinfo(parent.dtype).max
    valid_parent = parent >= 0
    ordered_parent, order = torch.sort(
        torch.where(valid_parent, parent, sentinel), dim=1,
    )
    query = child.clamp_min(0)
    location = torch.searchsorted(ordered_parent, query).clamp_max(
        parent.shape[1] - 1
    )
    matched = (
        (child >= 0)
        & (torch.gather(ordered_parent, 1, location) == query)
    )
    parent_location = torch.gather(order, 1, location)
    extra_dimensions = parent_value.ndim - 2
    gather_index = parent_location
    for _ in range(extra_dimensions):
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(
        *parent_location.shape, *parent_value.shape[2:]
    )
    mapped = torch.gather(parent_value, 1, gather_index)
    present = matched
    for _ in range(extra_dimensions):
        present = present.unsqueeze(-1)
    mapped = torch.where(
        present,
        mapped,
        torch.full_like(mapped, missing_value),
    )
    return mapped, matched


def _select_context(
    context: RequestContextBatch, selected: torch.Tensor,
) -> RequestContextBatch:
    return RequestContextBatch(
        request_id=context.request_id[selected],
        request_time=context.request_time[selected],
        user_event_counts=context.user_event_counts[selected],
        user_surface_counts=context.user_surface_counts[selected],
        history_item_id=context.history_item_id[selected],
        history_event_time=context.history_event_time[selected],
        history_ingest_time=context.history_ingest_time[selected],
        feature_as_of_ingest_time=context.feature_as_of_ingest_time[selected],
    )


class RequestLevelJoiner:
    def __init__(self, config: JoinerConfig, catalog: PublicCatalog):
        self.config = config
        self.catalog = catalog

    def materialize(
        self,
        trace: RequestCandidateTrace,
        context: RequestContextBatch,
        events: AppEventBatch,
        event_watermark: int,
    ) -> JoinedSampleAuthorities:
        if not torch.equal(trace.request_id, context.request_id):
            raise ValueError("request trace and feature context are misaligned")
        fine = self._fine(trace, context, events, event_watermark)
        coarse = self._coarse(trace, context, fine)
        recall = self._recall(trace, context, fine)
        return JoinedSampleAuthorities(
            recall=recall,
            coarse=coarse,
            fine=fine,
            event_watermark=event_watermark,
            manifest=trace.manifest,
        )

    def _fine(
        self,
        trace: RequestCandidateTrace,
        context: RequestContextBatch,
        events: AppEventBatch,
        watermark: int,
    ) -> FineRankExampleBatch:
        tasks = self.config.tasks
        labels = torch.stack(tuple(
            _event_values(events, task.event_type, trace) for task in tasks
        ), dim=2)
        item = trace.exposed_item_id.clamp_min(0)
        content_kind = self.catalog.content_kind[item]
        surface = trace.surface[:, None].expand_as(trace.exposed_item_id)
        valid = trace.exposed_item_id >= 0
        masks = []
        for task in tasks:
            mature = watermark >= trace.event_time + task.maturity_ticks
            masks.append(
                valid
                & mature[:, None]
                & _task_applicability(task, surface, content_kind)
            )
        label_mask = torch.stack(masks, dim=2)
        dwell = _event_values(
            events,
            EventType.DWELL,
            trace,
            values=events.duration_ms,
        )
        served_score, _ = _map_stage(
            trace.exposed_item_id,
            trace.fine_item_id,
            trace.fine_score,
            0.0,
        )
        return FineRankExampleBatch(
            request_id=trace.request_id,
            user_id=trace.user_id,
            surface=trace.surface,
            request_time=trace.event_time,
            item_id=trace.exposed_item_id,
            position=trace.exposed_position,
            served_score=served_score,
            exposure_probability=trace.exposure_probability,
            assignment_probability=trace.assignment_probability,
            recall_version_id=trace.recall_version_id,
            coarse_version_id=trace.coarse_version_id,
            fine_version_id=trace.fine_version_id,
            mix_version_id=trace.mix_version_id,
            labels=labels,
            label_mask=label_mask,
            dwell_ms=dwell,
            context=context,
            task_names=tuple(task.name for task in tasks),
            task_maturity_ticks=tuple(task.maturity_ticks for task in tasks),
        )

    def _coarse(
        self,
        trace: RequestCandidateTrace,
        context: RequestContextBatch,
        fine: FineRankExampleBatch,
    ) -> CoarseRankExampleBatch:
        route, _ = _map_stage(
            trace.coarse_item_id,
            trace.recall_item_id,
            trace.recall_route_id,
            -1,
        )
        teacher_score, teacher_mask = _map_stage(
            trace.coarse_item_id,
            trace.fine_item_id,
            trace.fine_score,
            0.0,
        )
        weights = torch.tensor(
            [task.weight for task in self.config.tasks],
            device=fine.labels.device,
        )
        exposed_value = (
            fine.labels * fine.label_mask.float() * weights
        ).sum(dim=2)
        hard_label, exposed = _map_stage(
            trace.coarse_item_id,
            fine.item_id,
            exposed_value,
            0.0,
        )
        observed, _ = _map_stage(
            trace.coarse_item_id,
            fine.item_id,
            fine.label_mask.any(dim=2),
            False,
        )
        return CoarseRankExampleBatch(
            request_id=trace.request_id,
            item_id=trace.coarse_item_id,
            route_id=route,
            served_score=trace.coarse_score,
            sampling_probability=trace.coarse_sampling_probability,
            hard_label=hard_label,
            hard_label_mask=exposed & observed,
            teacher_score=teacher_score,
            teacher_mask=teacher_mask,
            recall_version_id=trace.recall_version_id,
            coarse_version_id=trace.coarse_version_id,
            fine_version_id=trace.fine_version_id,
            context=context,
        )

    def _recall(
        self,
        trace: RequestCandidateTrace,
        context: RequestContextBatch,
        fine: FineRankExampleBatch,
    ) -> RecallExampleBatch:
        weights = torch.tensor(
            [task.weight for task in self.config.tasks],
            device=fine.labels.device,
        ).clamp_min(0.0)
        strength = (
            fine.labels * fine.label_mask.float() * weights
        ).sum(dim=2)
        positive_strength, position = strength.max(dim=1)
        selected = positive_strength > 0.0
        positive_item = torch.gather(
            fine.item_id, 1, position[:, None]
        ).squeeze(1)
        exposed_negative = (
            (fine.item_id >= 0)
            & (strength == 0.0)
            & fine.label_mask.any(dim=2)
        )
        exposed_pool = torch.where(
            exposed_negative, fine.item_id, torch.full_like(fine.item_id, -1),
        )
        recalled_is_exposed = (
            trace.recall_item_id[:, :, None] == fine.item_id[:, None, :]
        ).any(dim=2)
        recalled_pool = torch.where(
            (trace.recall_item_id >= 0) & ~recalled_is_exposed,
            trace.recall_item_id,
            torch.full_like(trace.recall_item_id, -1),
        )
        pool = torch.cat((exposed_pool, recalled_pool), dim=1)
        source = torch.cat((
            torch.zeros_like(exposed_pool),
            torch.ones_like(recalled_pool),
        ), dim=1)
        random = uniform_for_items(
            trace.request_id,
            pool.clamp_min(0),
            0,
            1_613,
            self.config.sampling_seed,
        ).masked_fill(pool < 0, float("inf"))
        order = torch.argsort(random, dim=1)
        width = min(self.config.recall_negatives, pool.shape[1])
        chosen = order[:, :width]
        negative = torch.gather(pool, 1, chosen)
        negative_source = torch.gather(source, 1, chosen)
        if width < self.config.recall_negatives:
            padding = self.config.recall_negatives - width
            negative = torch.nn.functional.pad(negative, (0, padding), value=-1)
            negative_source = torch.nn.functional.pad(
                negative_source, (0, padding), value=-1,
            )
        available = (pool >= 0).sum(dim=1)
        probability = (
            self.config.recall_negatives / available.clamp_min(1).float()
        ).clamp_max(1.0)
        negative_probability = torch.where(
            negative >= 0,
            probability[:, None].expand_as(negative),
            torch.zeros_like(negative, dtype=torch.float),
        )
        return RecallExampleBatch(
            request_id=trace.request_id[selected],
            user_id=trace.user_id[selected],
            surface=trace.surface[selected],
            positive_item_id=positive_item[selected],
            positive_strength=positive_strength[selected],
            recall_version_id=trace.recall_version_id[selected],
            negative_item_id=negative[selected],
            negative_source=negative_source[selected],
            negative_sampling_probability=negative_probability[selected],
            context=_select_context(context, selected),
        )
