"""Single-world event loop with one pre-period and forked policy arms."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

import torch

from .contracts import TwinConfig, TwinPolicy
from .environment.runtime import UserEnvironment
from .ledger import append_exposure
from .metrics import add_active_day, accumulate_metrics, empty_user_metrics
from .platform.updates import apply_daily_observations, apply_response_events
from .state import (
    TwinSnapshot,
    initialize_catalog_pair,
    initialize_user_pair,
    select_users,
    writeback_users,
)
from .environment.latent import select_latent, writeback_latent
from .exchange import ServedSlate
from .serving.surfaces import build_slate
from .serving.models import as_serving_stack
from .serving.trace import RequestTrace
from .world.context import advance_context, initialize_context
from .world.supply import accumulate_supply_feedback, advance_supply_day


@dataclass
class TwinRun:
    snapshot: TwinSnapshot | None
    user_metrics: tuple[torch.Tensor, ...]
    trace: RequestTrace
    supply_days: list[dict[str, int]]
    seconds: float
    users_per_second: float


def _serve_chunk(
    kernel, snapshot, policy, users, latent_users, metrics, trace, step,
    trace_limit,
):
    surface = kernel.environment.begin_step(
        users, latent_users, snapshot.context, step
    )
    candidates = build_slate(
        kernel.config, policy, users, snapshot.catalog,
        snapshot.context, surface, step,
    )
    response = kernel.environment.respond(
        users, latent_users, snapshot.catalog, snapshot.latent_catalog,
        snapshot.context, ServedSlate(candidates.exposed_item_ids), surface,
        step,
    )
    accumulate_metrics(metrics, response, surface)
    accumulate_supply_feedback(snapshot.catalog, response)
    if trace_limit > 0:
        trace.append(
            users, surface, candidates, response, step, trace_limit,
            as_serving_stack(policy).name, "single_world",
        )
    append_exposure(
        users.ledger, snapshot.catalog, response.selected_item,
        surface, step, response.active,
    )
    apply_response_events(users, snapshot.catalog, response, surface)
    kernel.environment.commit(
        users, latent_users, snapshot.latent_catalog,
        response, surface, step,
    )
    return min(trace_limit, int(response.active.sum()))


class DigitalTwinKernel:
    def __init__(self, config: TwinConfig):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA digital twin requested but unavailable")
        self.environment = UserEnvironment(config, self.device)

    def initialize(self) -> TwinSnapshot:
        pairs = tuple(
            initialize_user_pair(
                self.config, start,
                min(self.config.batch_users, self.config.users - start),
                self.device,
            )
            for start in range(0, self.config.users, self.config.batch_users)
        )
        users = tuple(pair[0] for pair in pairs)
        latent_users = tuple(pair[1] for pair in pairs)
        catalog, latent_catalog = initialize_catalog_pair(
            self.config, self.device
        )
        metrics = tuple(
            empty_user_metrics(len(batch.user_id), self.device)
            for batch in users
        )
        return TwinSnapshot(
            users=users,
            latent_users=latent_users,
            catalog=catalog,
            latent_catalog=latent_catalog,
            step=0,
            preperiod_user_metrics=metrics,
            context=initialize_context(self.config, self.device),
        )

    def advance_day(self, snapshot, metrics, day):
        supply = advance_supply_day(
            self.config, snapshot.catalog, snapshot.latent_catalog, day
        )
        for batch, latent, batch_metrics in zip(
            snapshot.users, snapshot.latent_users, metrics, strict=True
        ):
            add_active_day(batch_metrics, batch.active)
            apply_daily_observations(batch)
            lifecycle = self.environment.advance_day(batch, latent, day)
            for name, value in lifecycle.items():
                supply[name] = supply.get(name, 0) + value
        return supply

    @torch.inference_mode()
    def run(
        self,
        snapshot: TwinSnapshot,
        policy: TwinPolicy,
        steps: int,
        *,
        record_trace: bool = True,
        trace_limit: int | None = None,
    ) -> TwinRun:
        metrics = tuple(
            empty_user_metrics(len(batch.user_id), self.device)
            for batch in snapshot.users
        )
        trace = RequestTrace()
        trace_limit = trace_limit or self.config.audit_users
        chunks = sum(
            ceil(len(users.user_id) / self.config.serve_chunk_users)
            for users in snapshot.users
        )
        trace_chunk_limit = ceil(trace_limit / max(chunks, 1))
        supply_days = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        stop = snapshot.step + steps
        for step in range(snapshot.step, stop):
            advance_context(snapshot.context, self.config, step)
            trace_remaining = trace_limit if record_trace else 0
            for users, latent_users, batch_metrics in (
                zip(
                    snapshot.users, snapshot.latent_users, metrics,
                    strict=True,
                )
            ):
                for start in range(0, len(users.user_id), self.config.serve_chunk_users):
                    selector = slice(start, start + self.config.serve_chunk_users)
                    chunk = select_users(users, selector)
                    latent_chunk = select_latent(latent_users, selector)
                    chunk_metrics = batch_metrics[selector].clone()
                    consumed = _serve_chunk(
                        self, snapshot, policy, chunk, latent_chunk,
                        chunk_metrics, trace, step,
                        min(trace_remaining, trace_chunk_limit),
                    )
                    trace_remaining -= consumed
                    writeback_users(users, chunk, selector)
                    writeback_latent(latent_users, latent_chunk, selector)
                    batch_metrics[selector] = chunk_metrics
            if (step + 1) % self.config.steps_per_day == 0:
                supply_days.append(self.advance_day(
                    snapshot, metrics, (step + 1) // self.config.steps_per_day
                ))
        snapshot.step = stop
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = perf_counter() - started
        return TwinRun(
            snapshot=snapshot,
            user_metrics=metrics,
            trace=trace,
            supply_days=supply_days,
            seconds=seconds,
            users_per_second=(self.config.users * steps / max(seconds, 1e-9)),
        )

    def preperiod(self, policy: TwinPolicy) -> TwinRun:
        return self.preperiod_from(self.initialize(), policy)

    def preperiod_from(
        self, snapshot: TwinSnapshot, policy: TwinPolicy,
    ) -> TwinRun:
        run = self.run(snapshot, policy, self.config.preperiod_steps)
        if run.snapshot is None:
            raise RuntimeError("pre-period run lost its world snapshot")
        run.snapshot.preperiod_user_metrics = tuple(
            value.clone() for value in run.user_metrics
        )
        return run

    def arm(self, shared: TwinSnapshot, policy: TwinPolicy) -> TwinRun:
        return self.run(
            shared.fork(), policy, self.config.measurement_steps
        )
