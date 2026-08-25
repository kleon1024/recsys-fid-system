"""Source-aware recall negatives and sampled-softmax correction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from ...randomness.counter import uniform
from ..catalog import PublicCatalog


class NegativeSource(IntEnum):
    IN_BATCH = 0
    EXPOSED = 1
    MINED = 2
    CATALOG = 3


NEGATIVE_SOURCE_WEIGHTS = (0.60, 0.15, 0.10, 0.15)


def negative_source_counts(total: int) -> tuple[int, ...]:
    """Allocate a fixed draw budget by largest remainder."""
    if total <= 0:
        raise ValueError("negative draw count must be positive")
    raw = tuple(total * weight for weight in NEGATIVE_SOURCE_WEIGHTS)
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(raw)), key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


@dataclass(frozen=True)
class NegativeSampleBatch:
    item_id: torch.Tensor
    source: torch.Tensor
    sampling_probability: torch.Tensor
    expected_count: torch.Tensor
    observed: torch.Tensor
    false_negative_mask: torch.Tensor


def _draw_from_pool(
    request_id: torch.Tensor,
    pool: torch.Tensor,
    valid: torch.Tensor,
    draws: int,
    stream: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    requests = len(request_id)
    if draws == 0:
        empty_item = torch.empty(
            requests, 0, device=request_id.device, dtype=torch.long,
        )
        return empty_item, empty_item.float()
    count = valid.sum(dim=1)
    probability = valid.float() / count.clamp_min(1)[:, None]
    cdf = probability.cumsum(dim=1).contiguous()
    random = uniform(request_id, 0, stream, seed, draws).contiguous()
    location = torch.searchsorted(cdf, random).clamp_max(pool.shape[1] - 1)
    item = torch.gather(pool, 1, location)
    has_pool = count > 0
    item = torch.where(has_pool[:, None], item, torch.full_like(item, -1))
    same_item = (
        (pool[:, :, None] == item[:, None, :])
        & valid[:, :, None]
        & (item[:, None, :] >= 0)
    )
    item_probability = same_item.sum(dim=1).float() / count.clamp_min(1)[:, None]
    item_probability = torch.where(
        item >= 0, item_probability, torch.zeros_like(item_probability),
    )
    return item, item_probability


def _draw_in_batch(
    request_id: torch.Tensor,
    positive_item_id: torch.Tensor,
    draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    requests = len(request_id)
    if draws == 0:
        empty_item = torch.empty(
            requests, 0, device=request_id.device, dtype=torch.long,
        )
        return empty_item, empty_item.float()
    if requests <= 1:
        missing = torch.full(
            (requests, draws), -1, device=request_id.device, dtype=torch.long,
        )
        return missing, torch.zeros_like(missing, dtype=torch.float)
    random = uniform(request_id, 0, 1_701, seed, draws)
    peer = torch.floor(random * (requests - 1)).long()
    row = torch.arange(requests, device=request_id.device)[:, None]
    peer += peer >= row
    item = positive_item_id[peer]
    _, inverse, frequency = torch.unique(
        positive_item_id,
        return_inverse=True,
        return_counts=True,
    )
    item_frequency = frequency[inverse[peer]]
    same_as_own = item == positive_item_id[:, None]
    item_frequency = item_frequency - same_as_own.long()
    probability = item_frequency.float() / (requests - 1)
    return item, probability


def _draw_catalog(
    request_id: torch.Tensor,
    catalog: PublicCatalog,
    draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    requests = len(request_id)
    if draws == 0:
        empty_item = torch.empty(
            requests, 0, device=request_id.device, dtype=torch.long,
        )
        return empty_item, empty_item.float()
    active = catalog.item_id[catalog.active]
    if not len(active):
        raise ValueError("recall catalog has no active items")
    random = uniform(request_id, 0, 1_739, seed, draws)
    location = torch.floor(random * len(active)).long().clamp_max(len(active) - 1)
    item = active[location]
    probability = torch.full_like(item, 1.0 / len(active), dtype=torch.float)
    return item, probability


def build_recall_negatives(
    *,
    request_id: torch.Tensor,
    positive_item_id: torch.Tensor,
    exposed_item_id: torch.Tensor,
    exposed_negative: torch.Tensor,
    recall_item_id: torch.Tensor,
    recalled_unexposed: torch.Tensor,
    history_item_id: torch.Tensor,
    catalog: PublicCatalog,
    total: int,
    seed: int,
) -> NegativeSampleBatch:
    counts = negative_source_counts(total)
    requests = len(request_id)
    if requests == 0:
        empty_item = torch.empty(
            0, total, device=request_id.device, dtype=torch.long,
        )
        return NegativeSampleBatch(
            item_id=empty_item,
            source=empty_item.clone(),
            sampling_probability=empty_item.float(),
            expected_count=empty_item.float(),
            observed=empty_item.bool(),
            false_negative_mask=empty_item.bool(),
        )
    exposed_valid = exposed_negative & (exposed_item_id >= 0)
    recall_valid = recalled_unexposed & (recall_item_id >= 0)
    positive_topic = catalog.topic_id[positive_item_id]
    recall_topic = catalog.topic_id[recall_item_id.clamp_min(0)]
    mined_valid = recall_valid & (recall_topic == positive_topic[:, None])
    pools = (
        _draw_in_batch(request_id, positive_item_id, counts[0], seed),
        _draw_from_pool(
            request_id, exposed_item_id, exposed_valid, counts[1], 1_709, seed,
        ),
        _draw_from_pool(
            request_id, recall_item_id, mined_valid, counts[2], 1_721, seed,
        ),
        _draw_catalog(request_id, catalog, counts[3], seed),
    )
    items = torch.cat(tuple(pool[0] for pool in pools), dim=1)
    probabilities = torch.cat(tuple(pool[1] for pool in pools), dim=1)
    source_parts = []
    expected_parts = []
    for source, (count, (_, probability)) in enumerate(zip(counts, pools)):
        source_parts.append(torch.full(
            (requests, count), source, device=request_id.device,
            dtype=torch.long,
        ))
        expected_parts.append(probability * count)
    sources = torch.cat(source_parts, dim=1)
    expected = torch.cat(expected_parts, dim=1)
    valid = items >= 0
    sources = torch.where(valid, sources, torch.full_like(sources, -1))
    in_history = (
        (items[:, :, None] == history_item_id[:, None, :])
        & (history_item_id[:, None, :] >= 0)
    ).any(dim=2)
    false_negative = valid & (
        (items == positive_item_id[:, None]) | in_history
    )
    observed = valid & (sources == int(NegativeSource.EXPOSED))
    return NegativeSampleBatch(
        item_id=items,
        source=sources,
        sampling_probability=probabilities,
        expected_count=expected,
        observed=observed,
        false_negative_mask=false_negative,
    )


def corrected_sampled_softmax_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    negative_expected_count: torch.Tensor,
    negative_loss_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Correct sampled negatives; the always-present positive is unchanged."""
    if positive_logits.ndim != 1:
        raise ValueError("positive logits must be one-dimensional")
    shape = (len(positive_logits), negative_logits.shape[1])
    for name, value in (
        ("negative_logits", negative_logits),
        ("negative_expected_count", negative_expected_count),
        ("negative_loss_mask", negative_loss_mask),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} shape does not match sampled negatives")
    if negative_loss_mask.any() and (
        negative_expected_count[negative_loss_mask] <= 0.0
    ).any():
        raise ValueError("sampled-softmax expected counts must be positive")
    corrected = negative_logits - negative_expected_count.clamp_min(1e-12).log()
    corrected = corrected.masked_fill(~negative_loss_mask, -torch.inf)
    logits = torch.cat((positive_logits[:, None], corrected), dim=1)
    targets = torch.zeros(len(positive_logits), device=logits.device, dtype=torch.long)
    if reduction not in {"mean", "none"}:
        raise ValueError("sampled-softmax reduction must be mean or none")
    return torch.nn.functional.cross_entropy(
        logits, targets, reduction=reduction,
    )
