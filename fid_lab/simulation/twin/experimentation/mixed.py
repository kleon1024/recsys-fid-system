"""Online-style A/B where policy cells share one evolving ecosystem."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

import torch

from ...experimentation.assignment import assign_binary_torch
from ..contracts import TwinPolicy
from ..environment.latent import select_latent, writeback_latent
from ..kernel import DigitalTwinKernel
from ..metrics import empty_user_metrics
from ..serving.trace import RequestTrace
from ..state import TwinSnapshot, select_users, writeback_users as writeback_chunk
from ..world.context import advance_context
from .runtime import serve_cell, subset_users, writeback_users


@dataclass
class MixedTwinRun:
    snapshot: TwinSnapshot
    user_metrics: tuple[torch.Tensor, ...]
    assigned_treatment: tuple[torch.Tensor, ...]
    traces: dict[str, RequestTrace]
    supply_days: list[dict[str, int]]
    seconds: float
    users_per_second: float


def _serve_chunk(
    kernel, snapshot, users, latent_users, batch_metrics, assigned, traces,
    control, treatment, step, trace_remaining, trace_chunk_limit,
):
    surface = kernel.environment.begin_step(
        users, latent_users, snapshot.context, step
    )
    for name, mask, policy in (
        ("control", ~assigned, control),
        ("treatment", assigned, treatment),
    ):
        if not mask.any():
            continue
        subset = subset_users(users, mask)
        latent_subset = select_latent(latent_users, mask)
        subset_metrics = batch_metrics[mask].clone()
        cell_limit = min(trace_remaining[name], trace_chunk_limit[name])
        serve_cell(
            kernel, snapshot, subset, latent_subset, surface[mask], policy,
            subset_metrics, traces[name], step, name, cell_limit,
        )
        trace_remaining[name] -= min(
            cell_limit, int(subset.active.sum())
        )
        batch_metrics[mask] = subset_metrics
        writeback_users(users, subset, mask)
        writeback_latent(latent_users, latent_subset, mask)


def _serve_batch(
    kernel, snapshot, users, latent_users, batch_metrics, assigned, traces,
    control, treatment, step, trace_remaining, trace_chunk_limit,
):
    size = kernel.config.serve_chunk_users
    for start in range(0, len(users.user_id), size):
        selector = slice(start, start + size)
        chunk = select_users(users, selector)
        latent_chunk = select_latent(latent_users, selector)
        chunk_metrics = batch_metrics[selector].clone()
        _serve_chunk(
            kernel, snapshot, chunk, latent_chunk, chunk_metrics,
            assigned[selector], traces, control, treatment, step,
            trace_remaining, trace_chunk_limit,
        )
        writeback_chunk(users, chunk, selector)
        writeback_latent(latent_users, latent_chunk, selector)
        batch_metrics[selector] = chunk_metrics


@torch.inference_mode()
def run_mixed_world_ab(
    kernel: DigitalTwinKernel,
    shared: TwinSnapshot,
    control: TwinPolicy,
    treatment: TwinPolicy,
    experiment_salt: int,
    trace_limit: int | None = None,
) -> MixedTwinRun:
    config = kernel.config
    snapshot = shared.fork()
    metrics = tuple(
        empty_user_metrics(len(batch.user_id), kernel.device)
        for batch in snapshot.users
    )
    assignments = tuple(
        assign_binary_torch(batch.user_id, experiment_salt)
        for batch in snapshot.users
    )
    traces = {"control": RequestTrace(), "treatment": RequestTrace()}
    trace_limit = trace_limit or config.audit_users
    chunks = sum(
        ceil(len(users.user_id) / config.serve_chunk_users)
        for users in snapshot.users
    )
    control_target = trace_limit // 2
    treatment_target = trace_limit - control_target
    trace_chunk_limit = {
        "control": ceil(control_target / max(chunks, 1)),
        "treatment": ceil(treatment_target / max(chunks, 1)),
    }
    supply_days = []
    if kernel.device.type == "cuda":
        torch.cuda.synchronize(kernel.device)
    started = perf_counter()
    stop = snapshot.step + config.measurement_steps
    for step in range(snapshot.step, stop):
        advance_context(snapshot.context, config, step)
        trace_remaining = {
            "control": trace_limit // 2,
            "treatment": trace_limit - trace_limit // 2,
        }
        for users, latent_users, batch_metrics, assigned in zip(
            snapshot.users, snapshot.latent_users, metrics, assignments,
            strict=True,
        ):
            _serve_batch(
                kernel, snapshot, users, latent_users, batch_metrics,
                assigned, traces, control, treatment, step,
                trace_remaining, trace_chunk_limit,
            )
        if (step + 1) % config.steps_per_day == 0:
            supply_days.append(kernel.advance_day(
                snapshot, metrics, (step + 1) // config.steps_per_day
            ))
    snapshot.step = stop
    if kernel.device.type == "cuda":
        torch.cuda.synchronize(kernel.device)
    seconds = perf_counter() - started
    return MixedTwinRun(
        snapshot=snapshot,
        user_metrics=metrics,
        assigned_treatment=assignments,
        traces=traces,
        supply_days=supply_days,
        seconds=seconds,
        users_per_second=(config.users * config.measurement_steps / seconds),
    )
