"""Tensorized candidate merging operators shared by retrieval policies."""

from __future__ import annotations

import torch


def reciprocal_rank_fusion(
    route_item: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    reciprocal_rank_constant: float,
    merged_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge route candidates while preserving route provenance bits."""
    requests, routes, route_k = route_item.shape
    rank = torch.arange(1, route_k + 1, device=route_item.device).float()
    contribution = 1.0 / (reciprocal_rank_constant + rank)
    score = contribution[None, None].expand(requests, routes, route_k)
    score = score * route_valid.float()
    bit = (
        2 ** torch.arange(routes, device=route_item.device, dtype=torch.long)
    )[None, :, None].expand_as(route_item) * route_valid.long()
    sentinel = torch.iinfo(torch.long).max
    flat_item = torch.where(
        route_valid, route_item, torch.full_like(route_item, sentinel),
    ).reshape(requests, -1)
    ordered_item, order = torch.sort(flat_item, dim=1)
    ordered_score = torch.gather(score.reshape(requests, -1), 1, order)
    ordered_bit = torch.gather(bit.reshape(requests, -1), 1, order)
    starts = torch.ones_like(ordered_item, dtype=torch.bool)
    starts[:, 1:] = ordered_item[:, 1:] != ordered_item[:, :-1]
    group = torch.cumsum(starts.long(), dim=1) - 1
    width = ordered_item.shape[1]
    merged_item = torch.full_like(ordered_item, -1)
    merged_score = torch.zeros(requests, width, device=route_item.device)
    merged_bit = torch.zeros_like(ordered_item)
    merged_item.scatter_(1, group, ordered_item)
    merged_score.scatter_add_(1, group, ordered_score)
    merged_bit.scatter_add_(1, group, ordered_bit)
    valid = merged_item != sentinel
    merged_score.masked_fill_(~valid, -torch.inf)
    keep = min(merged_k, width)
    position = torch.topk(merged_score, keep, dim=1).indices
    item = torch.gather(merged_item, 1, position)
    value = torch.gather(merged_score, 1, position)
    bits = torch.gather(merged_bit, 1, position)
    item = torch.where(torch.isfinite(value), item, torch.full_like(item, -1))
    if keep < merged_k:
        padding = merged_k - keep
        item = torch.nn.functional.pad(item, (0, padding), value=-1)
        value = torch.nn.functional.pad(value, (0, padding), value=-torch.inf)
        bits = torch.nn.functional.pad(bits, (0, padding), value=0)
    return item, value, bits
