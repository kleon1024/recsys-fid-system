"""Watermark-aware materializer for recall, coarse and fine authorities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

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
from .negative_sampling import build_recall_negatives


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
    short_sequence_length: int = 16
    sampling_seed: int = 1_607

    def __post_init__(self):
        if (
            self.ticks_per_day <= 0
            or self.recall_negatives <= 0
            or self.short_sequence_length <= 0
        ):
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
            LabelTask("search_success", EventType.SEARCH_SUCCESS, 0, 0.55),
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
    history_length = state.user_history_item.shape[1]
    event_number = (
        state.user_history_cursor[user, None]
        - history_length
        + torch.arange(history_length, device=user.device)[None, :]
    )
    history_valid = event_number >= 0
    history_slot = torch.remainder(event_number.clamp_min(0), history_length)

    def history(field: str, missing: float | int) -> torch.Tensor:
        values = torch.gather(getattr(state, field)[user], 1, history_slot)
        return torch.where(
            history_valid, values, torch.full_like(values, missing),
        ).clone()

    return RequestContextBatch(
        request_id=trace.request_id,
        request_time=trace.event_time,
        user_event_counts=state.user_event_counts[user].clone(),
        user_surface_counts=state.user_surface_counts[user].clone(),
        history_item_id=history("user_history_item", -1),
        history_event_type=history("user_history_event_type", -1),
        history_surface=history("user_history_surface", -1),
        history_duration_ms=history("user_history_duration_ms", 0.0),
        history_event_time=history("user_history_event_time", -1),
        history_ingest_time=history("user_history_ingest_time", -1),
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
    if event_type == EventType.PUBLISH:
        event_id = deterministic_event_id(
            events.request_id[selected],
            events.event_type[selected],
            events.source_candidate_id[selected],
            events.position[selected],
        )
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
    if task.event_type == EventType.SEARCH_SUCCESS:
        return surface == int(Surface.SEARCH)
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
    return context.select(selected)


def _map_fine_features(
    trace: RequestCandidateTrace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense, _ = _map_stage(
        trace.coarse_item_id,
        trace.recall_item_id,
        trace.candidate_dense_features,
        0.0,
    )
    fids, _ = _map_stage(
        trace.coarse_item_id,
        trace.recall_item_id,
        trace.candidate_sparse_fids,
        0,
    )
    buckets, _ = _map_stage(
        trace.coarse_item_id,
        trace.recall_item_id,
        trace.candidate_sparse_buckets,
        0,
    )
    return dense, fids, buckets


@dataclass(frozen=True)
class _FineLabelTensors:
    labels: torch.Tensor
    exposed: torch.Tensor
    applicable: torch.Tensor
    mature: torch.Tensor
    mask: torch.Tensor
    maturity_time: torch.Tensor
    dwell_ms: torch.Tensor


def _fine_label_tensors(
    trace: RequestCandidateTrace,
    events: AppEventBatch,
    catalog: PublicCatalog,
    tasks: tuple[LabelTask, ...],
    watermark: int,
) -> _FineLabelTensors:
    exposed_labels = torch.stack(tuple(
        _event_values(events, task.event_type, trace) for task in tasks
    ), dim=2)
    labels, _ = _map_stage(
        trace.coarse_item_id,
        trace.exposed_item_id,
        exposed_labels,
        0.0,
    )
    exposed, _ = _map_stage(
        trace.coarse_item_id,
        trace.exposed_item_id,
        torch.ones_like(trace.exposed_item_id, dtype=torch.bool),
        False,
    )
    item = trace.coarse_item_id.clamp_min(0)
    content_kind = catalog.content_kind[item]
    surface = trace.surface[:, None].expand_as(trace.coarse_item_id)
    valid = trace.coarse_item_id >= 0
    applicability = []
    maturity = []
    maturity_time = []
    for task in tasks:
        task_maturity_time = trace.event_time + task.maturity_ticks
        applicability.append(
            exposed & _task_applicability(task, surface, content_kind)
        )
        maturity.append(exposed & (watermark >= task_maturity_time)[:, None])
        maturity_time.append(task_maturity_time[:, None].expand_as(valid))
    applicable = torch.stack(applicability, dim=2)
    mature = torch.stack(maturity, dim=2)
    dwell, _ = _map_stage(
        trace.coarse_item_id,
        trace.exposed_item_id,
        _event_values(
            events, EventType.DWELL, trace, values=events.duration_ms,
        ),
        0.0,
    )
    return _FineLabelTensors(
        labels=labels,
        exposed=exposed,
        applicable=applicable,
        mature=mature,
        mask=applicable & mature,
        maturity_time=torch.stack(maturity_time, dim=2),
        dwell_ms=dwell,
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
        label = _fine_label_tensors(
            trace, events, self.catalog, tasks, watermark,
        )
        fine_admitted, _ = _map_stage(
            trace.coarse_item_id,
            trace.fine_item_id,
            torch.ones_like(trace.fine_item_id, dtype=torch.bool),
            False,
        )
        valid = trace.coarse_item_id >= 0
        recall_route, _ = _map_stage(
            trace.coarse_item_id,
            trace.recall_item_id,
            trace.recall_route_id,
            -1,
        )
        recall_score, _ = _map_stage(
            trace.coarse_item_id,
            trace.recall_item_id,
            trace.recall_score,
            0.0,
        )
        coarse_score, _ = _map_stage(
            trace.coarse_item_id,
            trace.recall_item_id,
            trace.coarse_input_score,
            0.0,
        )
        position, _ = _map_stage(
            trace.coarse_item_id,
            trace.exposed_item_id,
            trace.exposed_position,
            -1,
        )
        exposure_probability, _ = _map_stage(
            trace.coarse_item_id,
            trace.exposed_item_id,
            trace.exposure_probability,
            0.0,
        )
        candidate_exposure_probability, _ = _map_stage(
            trace.coarse_item_id,
            trace.recall_item_id,
            trace.candidate_exposure_probability,
            0.0,
        )
        dense_features, sparse_fids, sparse_buckets = _map_fine_features(trace)
        joint_probability = (
            exposure_probability
            * trace.assignment_probability[:, None]
        )
        return FineRankExampleBatch(
            request_id=trace.request_id,
            user_id=trace.user_id,
            surface=trace.surface,
            request_time=trace.event_time,
            item_id=trace.coarse_item_id,
            position=position,
            served_score=trace.fine_input_score,
            recall_route_id=recall_route,
            recall_score=recall_score,
            coarse_score=coarse_score,
            fine_admitted=fine_admitted,
            exposed=label.exposed,
            exposure_probability=exposure_probability,
            candidate_exposure_probability=candidate_exposure_probability,
            selection_policy_kind=trace.selection_policy_kind,
            exploration_rate=trace.exploration_rate,
            slate_log_probability=trace.slate_log_probability,
            assignment_probability=trace.assignment_probability,
            joint_logging_probability=joint_probability,
            randomized_support=(
                valid
                & (trace.exploration_rate[:, None] > 0.0)
                & (candidate_exposure_probability > 0.0)
            ),
            recall_version_id=trace.recall_version_id,
            coarse_version_id=trace.coarse_version_id,
            fine_version_id=trace.fine_version_id,
            mix_version_id=trace.mix_version_id,
            labels=label.labels,
            label_applicable=label.applicable,
            label_mature=label.mature,
            label_mask=label.mask,
            label_maturity_time=label.maturity_time,
            dwell_ms=label.dwell_ms,
            dense_features=dense_features,
            sparse_fids=sparse_fids,
            sparse_buckets=sparse_buckets,
            context=context,
            task_names=tuple(task.name for task in tasks),
            task_maturity_ticks=tuple(task.maturity_ticks for task in tasks),
            short_sequence_length=min(
                self.config.short_sequence_length,
                context.history_item_id.shape[1],
            ),
            feature_manifest_hash=trace.manifest.feature_manifest_hash,
        )

    def _coarse(
        self,
        trace: RequestCandidateTrace,
        context: RequestContextBatch,
        fine: FineRankExampleBatch,
    ) -> CoarseRankExampleBatch:
        teacher_score, teacher_mask = _map_stage(
            trace.recall_item_id,
            fine.item_id,
            fine.served_score,
            0.0,
        )
        valid_teacher = fine.item_id >= 0
        teacher_order = torch.argsort(
            fine.served_score.masked_fill(~valid_teacher, -torch.inf),
            dim=1,
            descending=True,
            stable=True,
        )
        fine_rank = torch.full_like(fine.item_id, -1)
        ordinal = torch.arange(
            1, fine.item_id.shape[1] + 1, device=fine.item_id.device,
        )[None].expand_as(fine.item_id)
        fine_rank.scatter_(1, teacher_order, ordinal)
        fine_rank = torch.where(valid_teacher, fine_rank, torch.full_like(fine_rank, -1))
        teacher_rank, _ = _map_stage(
            trace.recall_item_id,
            fine.item_id,
            fine_rank,
            -1,
        )
        selected_rank = torch.arange(
            1, trace.coarse_item_id.shape[1] + 1, device=trace.coarse_item_id.device,
        )[None].expand_as(trace.coarse_item_id)
        coarse_rank, coarse_admitted = _map_stage(
            trace.recall_item_id,
            trace.coarse_item_id,
            selected_rank,
            -1,
        )
        conflict = teacher_mask & coarse_admitted & (teacher_rank != coarse_rank)
        weights = torch.tensor(
            [task.weight for task in self.config.tasks],
            device=fine.labels.device,
        )
        exposed_value = (
            fine.labels * fine.label_mask.float() * weights
        ).sum(dim=2)
        hard_label, _ = _map_stage(
            trace.recall_item_id,
            fine.item_id,
            exposed_value,
            0.0,
        )
        observed, _ = _map_stage(
            trace.recall_item_id,
            fine.item_id,
            fine.label_mask.any(dim=2),
            False,
        )
        return CoarseRankExampleBatch(
            request_id=trace.request_id,
            item_id=trace.recall_item_id,
            route_id=trace.recall_route_id,
            recall_score=trace.recall_score,
            served_score=trace.coarse_input_score,
            coarse_admitted=coarse_admitted,
            coarse_rank=coarse_rank,
            admission_probability=trace.coarse_admission_probability,
            hard_label=hard_label,
            hard_label_mask=observed,
            teacher_score=teacher_score,
            teacher_rank=teacher_rank,
            teacher_mask=teacher_mask,
            conflict_mask=conflict,
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
        recalled_is_exposed = (
            (
                trace.recall_item_id[:, :, None]
                == fine.item_id[:, None, :]
            ) & fine.exposed[:, None, :]
        ).any(dim=2)
        selected_context = _select_context(context, selected)
        negatives = build_recall_negatives(
            request_id=trace.request_id[selected],
            positive_item_id=positive_item[selected],
            exposed_item_id=fine.item_id[selected],
            exposed_negative=exposed_negative[selected],
            recall_item_id=trace.recall_item_id[selected],
            recalled_unexposed=(
                (trace.recall_item_id >= 0) & ~recalled_is_exposed
            )[selected],
            history_item_id=selected_context.history_item_id,
            catalog=self.catalog,
            total=self.config.recall_negatives,
            seed=self.config.sampling_seed,
        )
        positive_route, _ = _map_stage(
            positive_item[:, None],
            trace.recall_item_id,
            trace.recall_route_id,
            -1,
        )
        positive_probability, _ = _map_stage(
            positive_item[:, None],
            trace.recall_item_id,
            trace.recall_sampling_probability,
            0.0,
        )
        return RecallExampleBatch(
            request_id=trace.request_id[selected],
            user_id=trace.user_id[selected],
            surface=trace.surface[selected],
            query_topic=trace.query_topic[selected],
            positive_item_id=positive_item[selected],
            positive_strength=positive_strength[selected],
            positive_route_id=positive_route.squeeze(1)[selected],
            positive_proposal_probability=(
                positive_probability.squeeze(1)[selected]
            ),
            recall_version_id=trace.recall_version_id[selected],
            negative_item_id=negatives.item_id,
            negative_source=negatives.source,
            negative_sampling_probability=negatives.sampling_probability,
            negative_expected_count=negatives.expected_count,
            negative_observed=negatives.observed,
            negative_false_negative_mask=negatives.false_negative_mask,
            context=selected_context,
            catalog_version=trace.manifest.catalog_version,
        )
