"""One-slot randomized support for new short-video item cold start."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....randomness.counter import uniform_for_items
from ...contracts import ContentKind, PlatformRequestBatch, Surface
from ..exploration import exploration_mask
from ..lifecycle import ContentLifecycle
from ..projection import PlatformProjectionState


@dataclass(frozen=True)
class ColdStartDraw:
    item: torch.Tensor
    eligible: torch.Tensor
    eligible_count: torch.Tensor
    randomized: torch.Tensor


def draw_cold_start_item(
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    content_kind: torch.Tensor,
    recall_item: torch.Tensor,
    recall_route_id: torch.Tensor,
    deterministic_exposed: torch.Tensor,
    *,
    rate: float,
    seed: int,
    cold_route_bit: int,
) -> ColdStartDraw:
    safe = recall_item.clamp_min(0)
    already_exposed = (
        safe[:, :, None] == deterministic_exposed[:, None, :]
    ).any(dim=2)
    eligible = (
        (recall_item >= 0)
        & (state.item_lifecycle[safe] == int(ContentLifecycle.COLD_START))
        & (content_kind[safe] == int(ContentKind.SHORT_VIDEO))
        & ((recall_route_id & cold_route_bit) > 0)
        & ~already_exposed
        & (requests.surface == int(Surface.FEED))[:, None]
    )
    key = uniform_for_items(
        requests.request_id,
        safe,
        requests.event_time[:, None],
        2_401,
        seed,
    ).masked_fill(~eligible, torch.inf)
    location = torch.argmin(key, dim=1)
    item = torch.gather(recall_item, 1, location[:, None]).squeeze(1)
    eligible_count = eligible.sum(dim=1)
    item = torch.where(
        eligible_count > 0, item, torch.full_like(item, -1),
    )
    randomized = exploration_mask(
        requests.request_id,
        requests.event_time,
        rate,
        seed,
    ) & (eligible_count > 0)
    return ColdStartDraw(item, eligible, eligible_count, randomized)


def inject_last(
    deterministic: torch.Tensor,
    item: torch.Tensor,
    randomized: torch.Tensor,
) -> torch.Tensor:
    selected = deterministic.clone()
    present = (selected == item[:, None]).any(dim=1)
    inject = randomized & ~present
    selected[inject, -1] = item[inject]
    return selected


def targeted_admission_probability(
    parent_item: torch.Tensor,
    deterministic_selected: torch.Tensor,
    draw: ColdStartDraw,
    rate: float,
) -> torch.Tensor:
    valid = parent_item >= 0
    supported = draw.eligible_count > 0
    effective_rate = rate * supported.float()
    deterministic = (
        parent_item[:, :, None] == deterministic_selected[:, None, :]
    ).any(dim=2)
    eligible_outside = draw.eligible & ~deterministic
    outside_fraction = (
        eligible_outside.sum(dim=1).float()
        / draw.eligible_count.clamp_min(1).float()
    )
    displaced = parent_item == deterministic_selected[:, -1, None]
    probability = deterministic.float() * (
        1.0
        - effective_rate[:, None]
        * outside_fraction[:, None]
        * displaced.float()
    )
    probability += (
        effective_rate / draw.eligible_count.clamp_min(1).float()
    )[:, None] * eligible_outside.float()
    return probability.masked_fill(~valid, 0.0)


def targeted_position_probability(
    selected_item: torch.Tensor,
    deterministic_item: torch.Tensor,
    draw: ColdStartDraw,
    rate: float,
) -> torch.Tensor:
    probability = torch.ones_like(selected_item, dtype=torch.float)
    supported = draw.eligible_count > 0
    last_is_deterministic = selected_item[:, -1] == deterministic_item[:, -1]
    deterministic_probability = 1.0 - rate * supported.float()
    randomized_probability = (
        rate * supported.float()
        / draw.eligible_count.clamp_min(1).float()
    )
    probability[:, -1] = torch.where(
        last_is_deterministic,
        deterministic_probability,
        randomized_probability,
    )
    return probability.masked_fill(selected_item < 0, 0.0)


def targeted_slate_log_probability(
    selected_item: torch.Tensor,
    deterministic_item: torch.Tensor,
    draw: ColdStartDraw,
    rate: float,
) -> torch.Tensor:
    supported = draw.eligible_count > 0
    deterministic = (selected_item == deterministic_item).all(dim=1)
    probability = torch.where(
        deterministic,
        1.0 - rate * supported.float(),
        rate * supported.float() / draw.eligible_count.clamp_min(1).float(),
    )
    return probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
