"""Deterministic, partition-invariant exploration for the factual cascade."""

from __future__ import annotations

import torch

from ...randomness.counter import uniform, uniform_for_items


def exploration_mask(
    request_id: torch.Tensor,
    event_time: torch.Tensor,
    rate: float,
    seed: int,
) -> torch.Tensor:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("exploration rate must be in [0, 1]")
    if rate == 0.0:
        return torch.zeros_like(request_id, dtype=torch.bool)
    draw = uniform(request_id, event_time, 2_311, seed)
    return draw < rate


def random_ordered_top(
    request_id: torch.Tensor,
    item_id: torch.Tensor,
    limit: int,
    seed: int,
) -> torch.Tensor:
    if limit <= 0:
        raise ValueError("random top limit must be positive")
    valid = item_id >= 0
    key = uniform_for_items(request_id, item_id.clamp_min(0), 0, 2_313, seed)
    key = key.masked_fill(~valid, torch.inf)
    width = min(limit, item_id.shape[1])
    order = torch.argsort(key, dim=1, stable=True)[:, :width]
    selected = torch.gather(item_id, 1, order)
    if width < limit:
        selected = torch.nn.functional.pad(selected, (0, limit - width), value=-1)
    return selected


def mixture_position_probability(
    selected_item: torch.Tensor,
    deterministic_item: torch.Tensor,
    eligible_count: torch.Tensor,
    exploration_rate: float,
) -> torch.Tensor:
    if selected_item.shape != deterministic_item.shape:
        raise ValueError("selected and deterministic slates must align")
    rate = torch.full_like(eligible_count, exploration_rate, dtype=torch.float)
    random_probability = rate / eligible_count.clamp_min(1).float()
    deterministic_probability = (
        (1.0 - rate[:, None]) * (selected_item == deterministic_item).float()
    )
    probability = random_probability[:, None] + deterministic_probability
    return probability.masked_fill(selected_item < 0, 0.0)


def mixture_admission_probability(
    parent_item: torch.Tensor,
    deterministic_selected: torch.Tensor,
    exploration_rate: float,
) -> torch.Tensor:
    valid = parent_item >= 0
    eligible_count = valid.sum(dim=1)
    admitted_count = (deterministic_selected >= 0).sum(dim=1)
    random_probability = (
        exploration_rate
        * admitted_count.float()
        / eligible_count.clamp_min(1).float()
    )
    deterministic_admitted = (
        parent_item[:, :, None] == deterministic_selected[:, None, :]
    ).any(dim=2)
    probability = (
        random_probability[:, None]
        + (1.0 - exploration_rate) * deterministic_admitted.float()
    )
    return probability.masked_fill(~valid, 0.0)


def mixture_slate_log_probability(
    selected_item: torch.Tensor,
    deterministic_item: torch.Tensor,
    eligible_count: torch.Tensor,
    exploration_rate: float,
) -> torch.Tensor:
    valid_count = (selected_item >= 0).sum(dim=1).float()
    eligible = eligible_count.float()
    log_permutations = torch.lgamma(eligible + 1.0) - torch.lgamma(
        eligible - valid_count + 1.0,
    )
    rate = torch.full_like(eligible, exploration_rate)
    random_probability = rate * torch.exp(-log_permutations)
    deterministic_match = (selected_item == deterministic_item).all(dim=1)
    probability = random_probability + (1.0 - rate) * deterministic_match.float()
    empty = eligible_count == 0
    probability = torch.where(empty, torch.ones_like(probability), probability)
    return probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
