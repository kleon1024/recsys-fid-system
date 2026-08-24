"""Shared state-safe request serving primitives for experiment runtimes."""

from __future__ import annotations

import torch

from ..environment.latent import LatentUserState
from ..exchange import ServedSlate
from ..ledger import append_exposure
from ..metrics import accumulate_metrics
from ..platform.updates import apply_response_events
from ..serving.surfaces import build_slate
from ..serving.trace import RequestTrace
from ..state import (
    TwinSnapshot,
    UserState,
    select_users,
    writeback_users as writeback_selected_users,
)
from ..world.supply import accumulate_supply_feedback


def subset_users(users: UserState, mask: torch.Tensor) -> UserState:
    return select_users(users, mask)


def writeback_users(
    users: UserState, subset: UserState, mask: torch.Tensor,
) -> None:
    writeback_selected_users(users, subset, mask)


def serve_cell(
    kernel,
    snapshot: TwinSnapshot,
    users: UserState,
    latent_users: LatentUserState,
    surface: torch.Tensor,
    policy,
    metrics: torch.Tensor,
    trace: RequestTrace,
    step: int,
    experiment_cell: str,
    trace_limit: int,
) -> None:
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
    trace.append(
        users, surface, candidates, response, step, trace_limit,
        policy.name, experiment_cell,
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
