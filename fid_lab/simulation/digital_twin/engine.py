"""Two-phase kernel: all cells read one snapshot, then events commit once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch

from ..randomness.counter import uniform
from .contracts import AppEventBatch, PlatformRequestBatch, RenderedSlateBatch
from .event_log import ObservableEventLog


class EcosystemWorld(Protocol):
    def schedule(self, logical_time: int) -> AppEventBatch: ...

    def snapshot(self) -> object: ...

    def respond(
        self, snapshot: object, slate: RenderedSlateBatch
    ) -> AppEventBatch: ...

    def commit(self, events: AppEventBatch) -> None: ...


class RecommendationPlatform(Protocol):
    def ingest(self, events: AppEventBatch) -> None: ...

    def snapshot(self) -> object: ...

    def open_requests(
        self, entry_events: AppEventBatch
    ) -> PlatformRequestBatch: ...

    def render(
        self,
        snapshot: object,
        requests: PlatformRequestBatch,
        policy: object,
        experiment_cell: int,
        assignment_probability: torch.Tensor,
    ) -> RenderedSlateBatch: ...


@dataclass(frozen=True)
class ExperimentAssignment:
    """The one factual serving arm selected for each opened request."""

    cell_by_request: torch.Tensor
    probability_by_request: torch.Tensor
    policies: Mapping[int, object]
    analysis_cells: tuple[int, ...] = (0, 1)
    default_cell: int | None = None

    def __post_init__(self):
        if self.probability_by_request.shape != self.cell_by_request.shape:
            raise ValueError("assignment probability must align with requests")
        if len(self.probability_by_request) and (
            (self.probability_by_request <= 0.0)
            | (self.probability_by_request > 1.0)
        ).any():
            raise ValueError("assignment probability must be in (0, 1]")
        unknown = set(int(value) for value in torch.unique(self.cell_by_request))
        unknown -= set(self.policies)
        if unknown:
            raise ValueError(f"experiment has cells without policies: {unknown}")
        if not set(self.analysis_cells).issubset(self.policies):
            raise ValueError("analysis cells must have serving policies")
        if self.default_cell is not None:
            if self.default_cell not in self.policies:
                raise ValueError("default traffic cell must have a policy")
            if self.default_cell in self.analysis_cells:
                raise ValueError("default traffic cannot enter experiment analysis")

    @property
    def experiment_mask(self) -> torch.Tensor:
        mask = torch.zeros_like(self.cell_by_request, dtype=torch.bool)
        for cell in self.analysis_cells:
            mask |= self.cell_by_request == cell
        return mask


@dataclass(frozen=True)
class ExperimentPlan:
    """Stable online allocation rule, evaluated only after requests exist."""

    policies: Mapping[int, object]
    experiment_seed: int
    control_fraction: float
    treatment_fraction: float
    analysis_cells: tuple[int, ...] = (0, 1)
    default_cell: int = -1
    assignment_unit: str = "user"
    eligible_surfaces: tuple[int, ...] | None = None

    def __post_init__(self):
        if (
            self.control_fraction <= 0.0
            or self.treatment_fraction <= 0.0
            or self.control_fraction + self.treatment_fraction > 1.0
        ):
            raise ValueError("A/B fractions must be positive and sum to <= 1")
        required = {self.default_cell, *self.analysis_cells}
        if not required.issubset(self.policies):
            raise ValueError("default and analysis cells must have policies")
        if self.default_cell in self.analysis_cells:
            raise ValueError("default traffic cannot enter experiment analysis")
        if self.assignment_unit not in {"user", "request"}:
            raise ValueError("assignment unit must be user or request")

    @classmethod
    def ramped_user_ab(
        cls,
        *,
        active_policy: object,
        treatment_policy: object,
        experiment_seed: int,
        control_fraction: float,
        treatment_fraction: float,
        assignment_unit: str = "user",
        eligible_surfaces: tuple[int, ...] | None = None,
    ) -> ExperimentPlan:
        return cls(
            policies={-1: active_policy, 0: active_policy, 1: treatment_policy},
            experiment_seed=experiment_seed,
            control_fraction=control_fraction,
            treatment_fraction=treatment_fraction,
            assignment_unit=assignment_unit,
            eligible_surfaces=eligible_surfaces,
        )

    def assign(self, requests: PlatformRequestBatch) -> ExperimentAssignment:
        eligible = torch.ones_like(requests.user_id, dtype=torch.bool)
        if self.eligible_surfaces is not None:
            eligible.zero_()
            for surface in self.eligible_surfaces:
                eligible |= requests.surface == surface
        entity = (
            requests.user_id
            if self.assignment_unit == "user" else requests.request_id
        )
        draw = uniform(entity, 0, 811, self.experiment_seed)
        cell = torch.full_like(requests.user_id, -1)
        control = eligible & (draw < self.control_fraction)
        treatment = eligible & (draw >= self.control_fraction) & (
            draw < self.control_fraction + self.treatment_fraction
        )
        cell[control] = 0
        cell[treatment] = 1
        default_probability = 1.0 - (
            self.control_fraction + self.treatment_fraction
        )
        probability = torch.ones_like(draw)
        probability[eligible] = default_probability
        probability[control] = self.control_fraction
        probability[treatment] = self.treatment_fraction
        return ExperimentAssignment(
            cell_by_request=cell,
            probability_by_request=probability,
            policies=self.policies,
            analysis_cells=self.analysis_cells,
            default_cell=self.default_cell,
        )


@dataclass(frozen=True)
class TickResult:
    logical_time: int
    entry_events: AppEventBatch
    response_events: AppEventBatch
    rendered_requests: int
    experiment_requests: int
    baseline_requests: int
    cell_counts: dict[int, int]


class AtomicSimulationKernel:
    """Runs one event-time GPU microbatch against one factual world."""

    def __init__(
        self,
        world: EcosystemWorld,
        platform: RecommendationPlatform,
        event_log: ObservableEventLog,
    ) -> None:
        required_lateness = int(getattr(world, "max_reporting_lag", 0))
        if event_log.allowed_lateness < required_lateness:
            raise ValueError(
                "event log allowed lateness is below the world reporting lag"
            )
        self.world = world
        self.platform = platform
        self.event_log = event_log

    def step(
        self,
        logical_time: int,
        experiment: ExperimentPlan,
        *,
        cell_order: tuple[int, ...] | None = None,
    ) -> TickResult:
        entry = self.world.schedule(logical_time)
        if len(entry.ingest_time) and not (
            entry.ingest_time == logical_time
        ).all():
            raise ValueError("one step may ingest only its logical delivery time")
        if len(entry.event_time) and (entry.event_time > logical_time).any():
            raise ValueError("world cannot deliver events before they occur")
        self.event_log.validate(entry)
        self.world.commit(entry)
        self.event_log.append(entry)
        self.platform.ingest(entry)
        requests = self.platform.open_requests(entry)
        assignment = experiment.assign(requests)
        snapshot = self.world.snapshot()
        platform_snapshot = self.platform.snapshot()
        cells = cell_order or tuple(sorted(assignment.policies))
        if set(cells) != set(assignment.policies):
            raise ValueError("cell_order must contain every experiment cell once")
        proposals = []
        cell_counts = {}
        for cell in cells:
            selected = assignment.cell_by_request == cell
            cell_counts[cell] = int(selected.sum())
            if not selected.any():
                continue
            slate = self.platform.render(
                platform_snapshot, requests.select(selected),
                assignment.policies[cell], cell,
                assignment.probability_by_request[selected],
            )
            proposals.append(self.world.respond(snapshot, slate))
        response = AppEventBatch.concatenate(proposals)
        if len(response.ingest_time) and not (
            response.ingest_time == logical_time
        ).all():
            raise ValueError(
                "response events must ingest at the current logical time"
            )
        if len(response.event_time) and (response.event_time > logical_time).any():
            raise ValueError("response events cannot occur in the future")
        self.event_log.validate(response)
        self.world.commit(response)
        self.event_log.append(response)
        self.platform.ingest(response)
        return TickResult(
            logical_time=logical_time,
            entry_events=entry,
            response_events=response,
            rendered_requests=len(requests.request_id),
            experiment_requests=int(assignment.experiment_mask.sum()),
            baseline_requests=int((~assignment.experiment_mask).sum()),
            cell_counts=cell_counts,
        )
