"""Counter-based device RNG shared by partition-invariant simulators."""

from __future__ import annotations

import torch


MASK = 2_147_483_647


def _mix(keys):
    values = torch.bitwise_and(keys, MASK)
    values = torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))
    values = torch.bitwise_and(values * 73_415_833, MASK)
    values = torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))
    values = torch.bitwise_and(values * 73_415_833, MASK)
    return torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))


def uniform(entity_ids, step, stream, seed, width=None):
    keys = entity_ids.long()
    if width is not None:
        position = torch.arange(width, device=entity_ids.device, dtype=torch.long)
        keys = keys[:, None] * 1_103_515_245 + position[None, :] * 12_345
    values = _mix(
        keys * 48_271 + step * 7_919 + stream * 104_729 + seed * 503
    )
    result = (values.float() + 0.5) / (MASK + 1.0)
    return result.clamp_(max=1.0 - torch.finfo(result.dtype).eps)


def uniform_for_items(entity_ids, item_ids, step, stream, seed):
    entities = entity_ids.long()
    while entities.ndim < item_ids.ndim:
        entities = entities.unsqueeze(-1)
    values = _mix(
        entities * 1_103_515_245
        + item_ids.long() * 48_271
        + step * 7_919
        + stream * 104_729
        + seed * 503
    )
    result = (values.float() + 0.5) / (MASK + 1.0)
    return result.clamp_(max=1.0 - torch.finfo(result.dtype).eps)


def normal(entity_ids, step, stream, seed, width=None):
    first = uniform(entity_ids, step, stream, seed, width).clamp_min(1e-7)
    second = uniform(entity_ids, step, stream + 1, seed, width)
    return torch.sqrt(-2.0 * torch.log(first)) * torch.cos(
        2.0 * torch.pi * second
    )
