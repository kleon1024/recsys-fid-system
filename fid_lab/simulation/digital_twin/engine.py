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
    ) -> RenderedSlateBatch: ...


@dataclass(frozen=True)
class ExperimentPlan:
    cell_by_request: torch.Tensor
    policies: Mapping[int, object]
    analysis_cells: tuple[int, ...] = (0, 1)
    default_cell: int | None = None

    def __post_init__(self):
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

    @classmethod
    def ramped_user_ab(
        cls,
        requests: PlatformRequestBatch,
        *,
        active_policy: object,
        treatment_policy: object,
        experiment_seed: int,
        control_fraction: float,
        treatment_fraction: float,
        eligible: torch.Tensor | None = None,
    ) -> ExperimentPlan:
        if (
            control_fraction <= 0.0
            or treatment_fraction <= 0.0
            or control_fraction + treatment_fraction > 1.0
        ):
            raise ValueError("A/B fractions must be positive and sum to <= 1")
        eligible = (
            torch.ones_like(requests.user_id, dtype=torch.bool)
            if eligible is None else eligible.bool()
        )
        if eligible.shape != requests.user_id.shape:
            raise ValueError("eligibility must align with requests")
        draw = uniform(
            requests.user_id, 0, 811, experiment_seed
        )
        cell = torch.full_like(requests.user_id, -1)
        control = eligible & (draw < control_fraction)
        treatment = eligible & (draw >= control_fraction) & (
            draw < control_fraction + treatment_fraction
        )
        cell[control] = 0
        cell[treatment] = 1
        return cls(
            cell_by_request=cell,
            policies={
                -1: active_policy,
                0: active_policy,
                1: treatment_policy,
            },
            analysis_cells=(0, 1),
            default_cell=-1,
        )

    @property
    def experiment_mask(self) -> torch.Tensor:
        mask = torch.zeros_like(self.cell_by_request, dtype=torch.bool)
        for cell in self.analysis_cells:
            mask |= self.cell_by_request == cell
        return mask


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
    """Coordinates systems without exposing either side's private state."""

    def __init__(
        self,
        world: EcosystemWorld,
        platform: RecommendationPlatform,
        event_log: ObservableEventLog,
    ) -> None:
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
        self.world.commit(entry)
        self.event_log.append(entry)
        self.platform.ingest(entry)
        requests = self.platform.open_requests(entry)
        if len(requests.request_id) != len(experiment.cell_by_request):
            raise ValueError("experiment assignment does not match requests")
        snapshot = self.world.snapshot()
        platform_snapshot = self.platform.snapshot()
        cells = cell_order or tuple(sorted(experiment.policies))
        if set(cells) != set(experiment.policies):
            raise ValueError("cell_order must contain every experiment cell once")
        proposals = []
        cell_counts = {}
        for cell in cells:
            selected = experiment.cell_by_request == cell
            cell_counts[cell] = int(selected.sum())
            if not selected.any():
                continue
            slate = self.platform.render(
                platform_snapshot, requests.select(selected),
                experiment.policies[cell], cell,
            )
            proposals.append(self.world.respond(snapshot, slate))
        response = AppEventBatch.concatenate(proposals)
        self.world.commit(response)
        self.event_log.append(response)
        self.platform.ingest(response)
        return TickResult(
            logical_time=logical_time,
            entry_events=entry,
            response_events=response,
            rendered_requests=len(requests.request_id),
            experiment_requests=int(experiment.experiment_mask.sum()),
            baseline_requests=int((~experiment.experiment_mask).sum()),
            cell_counts=cell_counts,
        )
