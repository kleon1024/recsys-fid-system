"""Request-level serving trace and three non-interchangeable sample authorities."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..contracts import RenderedSlateBatch


def _aligned(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")


def _valid_items(items: torch.Tensor) -> torch.Tensor:
    return items >= 0


def _require_unique_per_request(name: str, items: torch.Tensor) -> None:
    if not items.shape[1]:
        raise ValueError(f"{name} candidate width must be positive")
    sentinel = torch.iinfo(items.dtype).max
    ordered = torch.sort(
        torch.where(_valid_items(items), items, sentinel), dim=1,
    ).values
    duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (
        ordered[:, 1:] != sentinel
    )
    if duplicate.any():
        raise ValueError(f"{name} contains duplicate candidates")


def _require_stage_subset(
    child_name: str,
    child: torch.Tensor,
    parent_name: str,
    parent: torch.Tensor,
) -> None:
    sentinel = torch.iinfo(parent.dtype).max
    ordered_parent = torch.sort(
        torch.where(_valid_items(parent), parent, sentinel), dim=1,
    ).values
    query = child.clamp_min(0)
    location = torch.searchsorted(ordered_parent, query).clamp_max(
        ordered_parent.shape[1] - 1
    )
    present = torch.gather(ordered_parent, 1, location) == query
    if (_valid_items(child) & ~present).any():
        raise ValueError(f"{child_name} is not a subset of {parent_name}")


@dataclass(frozen=True)
class TraceManifest:
    schema_version: str
    feature_version: str
    catalog_version: str
    policy_registry_version: str
    route_names: tuple[str, ...] = ()
    index_version: str = ""
    fid_version: str = ""
    lifecycle_version: str = ""


@dataclass(frozen=True)
class RequestCandidateTrace:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    event_time: torch.Tensor
    query_topic: torch.Tensor
    user_country: torch.Tensor
    user_region: torch.Tensor
    user_creator_id: torch.Tensor
    route_item_id: torch.Tensor
    route_score: torch.Tensor
    route_valid: torch.Tensor
    route_lifecycle_id: torch.Tensor
    recall_item_id: torch.Tensor
    recall_route_id: torch.Tensor
    recall_score: torch.Tensor
    recall_sampling_probability: torch.Tensor
    recall_lifecycle_id: torch.Tensor
    coarse_item_id: torch.Tensor
    coarse_score: torch.Tensor
    coarse_sampling_probability: torch.Tensor
    fine_item_id: torch.Tensor
    fine_score: torch.Tensor
    exposed_item_id: torch.Tensor
    exposed_position: torch.Tensor
    exposure_probability: torch.Tensor
    experiment_cell: torch.Tensor
    assignment_probability: torch.Tensor
    recall_version_id: torch.Tensor
    coarse_version_id: torch.Tensor
    fine_version_id: torch.Tensor
    mix_version_id: torch.Tensor
    manifest: TraceManifest

    def __post_init__(self):
        requests = len(self.request_id)
        for name in (
            "user_id",
            "surface",
            "event_time",
            "query_topic",
            "user_country",
            "user_region",
            "user_creator_id",
            "experiment_cell",
            "assignment_probability",
            "recall_version_id",
            "coarse_version_id",
            "fine_version_id",
            "mix_version_id",
        ):
            _aligned(name, getattr(self, name), (requests,))
        recall = self.recall_item_id.shape[1]
        if self.route_item_id.ndim != 3:
            raise ValueError("route candidates must be [request, route, rank]")
        route_shape = self.route_item_id.shape
        if route_shape[0] != requests:
            raise ValueError("route candidates do not align with requests")
        for name in ("route_score", "route_valid", "route_lifecycle_id"):
            _aligned(name, getattr(self, name), route_shape)
        if not torch.equal(self.route_valid, self.route_item_id >= 0):
            raise ValueError("route validity must match nonnegative route items")
        if self.manifest.route_names and (
            len(self.manifest.route_names) != route_shape[1]
        ):
            raise ValueError("route manifest does not match route candidates")
        coarse = self.coarse_item_id.shape[1]
        fine = self.fine_item_id.shape[1]
        exposed = self.exposed_item_id.shape[1]
        for name in (
            "recall_item_id",
            "recall_route_id",
            "recall_score",
            "recall_sampling_probability",
            "recall_lifecycle_id",
        ):
            _aligned(name, getattr(self, name), (requests, recall))
        for name in (
            "coarse_item_id",
            "coarse_score",
            "coarse_sampling_probability",
        ):
            _aligned(name, getattr(self, name), (requests, coarse))
        for name in ("fine_item_id", "fine_score"):
            _aligned(name, getattr(self, name), (requests, fine))
        for name in (
            "exposed_item_id",
            "exposed_position",
            "exposure_probability",
        ):
            _aligned(name, getattr(self, name), (requests, exposed))
        for name, items in (
            ("recall", self.recall_item_id),
            ("coarse", self.coarse_item_id),
            ("fine", self.fine_item_id),
            ("exposed", self.exposed_item_id),
        ):
            _require_unique_per_request(name, items)
        _require_stage_subset(
            "coarse", self.coarse_item_id, "recall", self.recall_item_id,
        )
        _require_stage_subset(
            "fine", self.fine_item_id, "coarse", self.coarse_item_id,
        )
        _require_stage_subset(
            "exposed", self.exposed_item_id, "fine", self.fine_item_id,
        )
        self._validate_probabilities()

    def _validate_probabilities(self) -> None:
        valid_recall = _valid_items(self.recall_item_id)
        valid_coarse = _valid_items(self.coarse_item_id)
        valid_exposed = _valid_items(self.exposed_item_id)
        for name, probability, valid in (
            (
                "recall_sampling_probability",
                self.recall_sampling_probability,
                valid_recall,
            ),
            (
                "coarse_sampling_probability",
                self.coarse_sampling_probability,
                valid_coarse,
            ),
            ("exposure_probability", self.exposure_probability, valid_exposed),
        ):
            if valid.any() and (
                (probability[valid] <= 0.0) | (probability[valid] > 1.0)
            ).any():
                raise ValueError(f"{name} must be in (0, 1] for valid items")
        if len(self.assignment_probability) and (
            (self.assignment_probability <= 0.0)
            | (self.assignment_probability > 1.0)
        ).any():
            raise ValueError("assignment_probability must be in (0, 1]")

    def select(self, selected: torch.Tensor) -> RequestCandidateTrace:
        values = {
            field.name: (
                self.manifest
                if field.name == "manifest"
                else getattr(self, field.name)[selected]
            )
            for field in fields(self)
        }
        return RequestCandidateTrace(**values)

    @classmethod
    def concatenate(
        cls, traces: tuple[RequestCandidateTrace, ...],
    ) -> RequestCandidateTrace:
        if not traces:
            raise ValueError("cannot concatenate an empty trace collection")
        manifest = traces[0].manifest
        if any(trace.manifest != manifest for trace in traces[1:]):
            raise ValueError("candidate trace manifests differ")
        values = {
            field.name: (
                manifest
                if field.name == "manifest"
                else torch.cat(tuple(
                    getattr(trace, field.name) for trace in traces
                ))
            )
            for field in fields(cls)
        }
        merged = cls(**values)
        order = torch.argsort(merged.request_id, stable=True)
        return merged.select(order)


@dataclass(frozen=True)
class RequestContextBatch:
    request_id: torch.Tensor
    request_time: torch.Tensor
    user_event_counts: torch.Tensor
    user_surface_counts: torch.Tensor
    history_item_id: torch.Tensor
    history_event_time: torch.Tensor
    history_ingest_time: torch.Tensor
    feature_as_of_ingest_time: torch.Tensor

    def __post_init__(self):
        requests = len(self.request_id)
        _aligned("request_time", self.request_time, (requests,))
        if self.user_event_counts.shape[0] != requests:
            raise ValueError("user event counters are not request-aligned")
        if self.user_surface_counts.shape[0] != requests:
            raise ValueError("user surface counters are not request-aligned")
        history = self.history_item_id.shape[1]
        for name in (
            "history_item_id",
            "history_event_time",
            "history_ingest_time",
        ):
            _aligned(name, getattr(self, name), (requests, history))
        _aligned(
            "feature_as_of_ingest_time",
            self.feature_as_of_ingest_time,
            (requests,),
        )
        if (self.feature_as_of_ingest_time > self.request_time).any():
            raise ValueError("request context contains future features")
        valid_history = self.history_item_id >= 0
        if valid_history.any() and (
            self.history_ingest_time[valid_history]
            > self.request_time[:, None].expand_as(self.history_item_id)[
                valid_history
            ]
        ).any():
            raise ValueError("request history contains future ingestion")

    def select(self, selected: torch.Tensor) -> RequestContextBatch:
        return RequestContextBatch(**{
            field.name: getattr(self, field.name)[selected]
            for field in fields(self)
        })

    @classmethod
    def concatenate(
        cls, contexts: tuple[RequestContextBatch, ...],
    ) -> RequestContextBatch:
        if not contexts:
            raise ValueError("cannot concatenate an empty context collection")
        merged = cls(**{
            field.name: torch.cat(tuple(
                getattr(context, field.name) for context in contexts
            ))
            for field in fields(cls)
        })
        order = torch.argsort(merged.request_id, stable=True)
        return merged.select(order)


@dataclass(frozen=True)
class ServingOutput:
    slate: RenderedSlateBatch
    candidate_trace: RequestCandidateTrace | None = None
    request_context: RequestContextBatch | None = None

    def __post_init__(self):
        if (self.candidate_trace is None) != (self.request_context is None):
            raise ValueError("candidate trace and request context must coexist")
        if self.candidate_trace is None:
            return
        trace = self.candidate_trace
        if not torch.equal(self.slate.request_id, trace.request_id):
            raise ValueError("serving slate and candidate trace requests differ")
        if not torch.equal(self.slate.item_ids, trace.exposed_item_id):
            raise ValueError("serving slate and trace exposures differ")
        if not torch.equal(self.slate.positions, trace.exposed_position):
            raise ValueError("serving slate and trace positions differ")


@dataclass(frozen=True)
class FineRankExampleBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    request_time: torch.Tensor
    item_id: torch.Tensor
    position: torch.Tensor
    served_score: torch.Tensor
    exposure_probability: torch.Tensor
    assignment_probability: torch.Tensor
    recall_version_id: torch.Tensor
    coarse_version_id: torch.Tensor
    fine_version_id: torch.Tensor
    mix_version_id: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    dwell_ms: torch.Tensor
    context: RequestContextBatch
    task_names: tuple[str, ...]
    task_maturity_ticks: tuple[int, ...]

    def __post_init__(self):
        if len(self.task_names) != self.labels.shape[2]:
            raise ValueError("fine task names do not match label width")
        if len(self.task_maturity_ticks) != len(self.task_names):
            raise ValueError("fine task maturity does not match task names")


@dataclass(frozen=True)
class CoarseRankExampleBatch:
    request_id: torch.Tensor
    item_id: torch.Tensor
    route_id: torch.Tensor
    served_score: torch.Tensor
    sampling_probability: torch.Tensor
    hard_label: torch.Tensor
    hard_label_mask: torch.Tensor
    teacher_score: torch.Tensor
    teacher_mask: torch.Tensor
    recall_version_id: torch.Tensor
    coarse_version_id: torch.Tensor
    fine_version_id: torch.Tensor
    context: RequestContextBatch


@dataclass(frozen=True)
class RecallExampleBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    positive_item_id: torch.Tensor
    positive_strength: torch.Tensor
    recall_version_id: torch.Tensor
    negative_item_id: torch.Tensor
    negative_source: torch.Tensor
    negative_sampling_probability: torch.Tensor
    context: RequestContextBatch


@dataclass(frozen=True)
class JoinedSampleAuthorities:
    recall: RecallExampleBatch
    coarse: CoarseRankExampleBatch
    fine: FineRankExampleBatch
    event_watermark: int
    manifest: TraceManifest
