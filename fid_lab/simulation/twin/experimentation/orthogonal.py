"""Orthogonal request experiments inside one shared evolving twin world."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import blake2b
from math import ceil
from time import perf_counter

import torch

from ...experimentation.assignment import validate_layer_ownership
from ...experimentation.contracts import ExperimentLayer
from ..contracts import TwinPolicy
from ..environment.latent import select_latent, writeback_latent
from ..kernel import DigitalTwinKernel
from ..metrics import empty_user_metrics
from ..serving.trace import RequestTrace
from ..state import (
    TwinSnapshot,
    UserState,
    select_users,
    writeback_users as writeback_chunk,
)
from ..world.context import advance_context
from .runtime import serve_cell, subset_users, writeback_users


class AssignmentUnit(str, Enum):
    USER = "user"
    REGION_TIME = "region_time"
    CREATOR_CLUSTER = "creator_cluster"


@dataclass(frozen=True)
class TwinLayerBinding:
    layer: ExperimentLayer
    unit: AssignmentUnit = AssignmentUnit.USER
    block_steps: int = 8

    def __post_init__(self):
        if self.block_steps < 1:
            raise ValueError("experiment switchback block must be positive")


@dataclass(frozen=True)
class ExperimentContext:
    signature: torch.Tensor
    assignments: tuple[torch.Tensor, ...]


def _assign_layer_torch(subject_ids, layer):
    mask = 0x7FFFFFFF
    salt = int.from_bytes(
        blake2b(layer.salt.encode(), digest_size=8, person=b"feed-ab").digest(),
        "big",
    ) & mask
    values = torch.bitwise_xor(subject_ids.long(), salt) & mask
    values = torch.bitwise_xor(values, values >> 16)
    values = (values * 0x045D9F3B) & mask
    values = torch.bitwise_xor(values, values >> 16)
    bucket = values.to(torch.float64) / float(mask + 1)
    allocations = [
        variant.allocation
        for experiment in layer.experiments
        for variant in experiment.variants
    ]
    thresholds = torch.tensor(
        allocations, device=subject_ids.device, dtype=torch.float64
    ).cumsum(dim=0)
    assignment = torch.bucketize(bucket, thresholds, right=True).long()
    assignment[assignment == len(allocations)] = -1
    return assignment


def _validate_policy_parameters(bindings: tuple[TwinLayerBinding, ...]) -> None:
    valid_fields = set(TwinPolicy.__dataclass_fields__)
    variants = (
        variant
        for binding in bindings
        for experiment in binding.layer.experiments
        for variant in experiment.variants
    )
    for variant in variants:
        unknown = set(variant.parameters) - valid_fields
        if unknown:
            raise ValueError(
                f"unknown twin policy parameters: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class TwinExperimentPlan:
    bindings: tuple[TwinLayerBinding, ...]

    def __post_init__(self):
        layers = tuple(binding.layer for binding in self.bindings)
        validate_layer_ownership(layers)
        _validate_policy_parameters(self.bindings)

    def _subject(
        self, binding: TwinLayerBinding, users: UserState, step: int,
    ) -> torch.Tensor:
        if binding.unit is AssignmentUnit.USER:
            return users.user_id
        if binding.unit is AssignmentUnit.REGION_TIME:
            block = step // binding.block_steps
            return users.region * 1_000_003 + block
        raise ValueError(
            "creator-cluster assignment is applied at the supply intervention, "
            "not before request candidates exist"
        )

    def assign(self, users: UserState, step: int) -> ExperimentContext:
        assignments = tuple(
            _assign_layer_torch(
                self._subject(binding, users, step), binding.layer
            )
            for binding in self.bindings
        )
        signature = torch.zeros_like(users.user_id)
        multiplier = 1
        for assignment, binding in zip(assignments, self.bindings, strict=True):
            cells = sum(
                len(experiment.variants)
                for experiment in binding.layer.experiments
            )
            signature += (assignment + 1) * multiplier
            multiplier *= cells + 1
        return ExperimentContext(signature, assignments)

    @property
    def cell_count(self) -> int:
        count = 1
        for binding in self.bindings:
            count *= 1 + sum(
                len(experiment.variants)
                for experiment in binding.layer.experiments
            )
        return count

    def resolve(self, baseline: TwinPolicy, signature: int) -> TwinPolicy:
        changes = {}
        remainder = signature
        labels = []
        for binding in self.bindings:
            variants = tuple(
                (experiment.name, variant)
                for experiment in binding.layer.experiments
                for variant in experiment.variants
            )
            radix = len(variants) + 1
            selected = remainder % radix - 1
            remainder //= radix
            if selected >= 0:
                experiment_name, variant = variants[selected]
                changes.update(variant.parameters)
                labels.append(f"{experiment_name}:{variant.name}")
        suffix = "+".join(labels) if labels else "all-control"
        return replace(baseline, name=f"{baseline.name}+{suffix}", **changes)


@dataclass
class OrthogonalTwinRun:
    snapshot: TwinSnapshot
    user_metrics: tuple[torch.Tensor, ...]
    traces: dict[int, RequestTrace]
    request_counts: dict[int, int]
    seconds: float


def _serve_chunk(
    kernel, snapshot, users, latent_users, metrics, plan, baseline, traces,
    request_counts, step, trace_remaining, trace_chunk_limit,
):
    surface = kernel.environment.begin_step(
        users, latent_users, snapshot.context, step
    )
    context = plan.assign(users, step)
    for signature in torch.unique(context.signature).tolist():
        mask = context.signature == signature
        request_counts[signature] = request_counts.get(signature, 0) + int(mask.sum())
        subset = subset_users(users, mask)
        latent_subset = select_latent(latent_users, mask)
        subset_metrics = metrics[mask].clone()
        trace = traces.setdefault(signature, RequestTrace())
        cell_limit = min(trace_remaining[signature], trace_chunk_limit)
        serve_cell(
            kernel, snapshot, subset, latent_subset, surface[mask],
            plan.resolve(baseline, signature), subset_metrics, trace, step,
            f"orthogonal:{signature}", cell_limit,
        )
        trace_remaining[signature] -= min(
            cell_limit, int(subset.active.sum())
        )
        metrics[mask] = subset_metrics
        writeback_users(users, subset, mask)
        writeback_latent(latent_users, latent_subset, mask)


def _serve_batch(
    kernel, snapshot, users, latent_users, metrics, plan, baseline, traces,
    request_counts, step, trace_remaining, trace_chunk_limit,
):
    size = kernel.config.serve_chunk_users
    for start in range(0, len(users.user_id), size):
        selector = slice(start, start + size)
        chunk = select_users(users, selector)
        latent_chunk = select_latent(latent_users, selector)
        chunk_metrics = metrics[selector].clone()
        _serve_chunk(
            kernel, snapshot, chunk, latent_chunk, chunk_metrics, plan,
            baseline, traces, request_counts, step, trace_remaining,
            trace_chunk_limit,
        )
        writeback_chunk(users, chunk, selector)
        writeback_latent(latent_users, latent_chunk, selector)
        metrics[selector] = chunk_metrics


@torch.inference_mode()
def run_orthogonal_world(
    kernel: DigitalTwinKernel,
    shared: TwinSnapshot,
    baseline: TwinPolicy,
    plan: TwinExperimentPlan,
    steps: int,
    trace_limit: int | None = None,
) -> OrthogonalTwinRun:
    snapshot = shared.fork()
    metrics = tuple(
        empty_user_metrics(len(batch.user_id), kernel.device)
        for batch in snapshot.users
    )
    traces: dict[int, RequestTrace] = {}
    request_counts: dict[int, int] = {}
    trace_limit = trace_limit or kernel.config.audit_users
    chunks = sum(
        ceil(len(users.user_id) / kernel.config.serve_chunk_users)
        for users in snapshot.users
    )
    started = perf_counter()
    stop = snapshot.step + steps
    for step in range(snapshot.step, stop):
        advance_context(snapshot.context, kernel.config, step)
        per_cell = max(trace_limit // plan.cell_count, 1)
        trace_chunk_limit = ceil(per_cell / max(chunks, 1))
        trace_remaining = {
            signature: per_cell for signature in range(plan.cell_count)
        }
        for users, latent_users, batch_metrics in zip(
            snapshot.users, snapshot.latent_users, metrics, strict=True
        ):
            _serve_batch(
                kernel, snapshot, users, latent_users, batch_metrics, plan,
                baseline, traces, request_counts, step, trace_remaining,
                trace_chunk_limit,
            )
        if (step + 1) % kernel.config.steps_per_day == 0:
            kernel.advance_day(
                snapshot, metrics, (step + 1) // kernel.config.steps_per_day
            )
    snapshot.step = stop
    return OrthogonalTwinRun(
        snapshot, metrics, traces, request_counts, perf_counter() - started
    )
