"""Counter-based device RNG invariant to batch partitioning."""

from __future__ import annotations

import torch


MASK = 2_147_483_647


def _mix(keys):
    """Avalanche 31-bit counters so event streams are not shifted copies."""
    values = torch.bitwise_and(keys, MASK)
    values = torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))
    values = torch.bitwise_and(values * 73_415_833, MASK)
    values = torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))
    values = torch.bitwise_and(values * 73_415_833, MASK)
    return torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))


def uniform(user_ids, step, stream, seed, width=None):
    keys = user_ids.long()
    if width is not None:
        position = torch.arange(width, device=user_ids.device, dtype=torch.long)
        keys = keys[:, None] * 1_103_515_245 + position[None, :] * 12_345
    values = _mix(
        keys * 48_271 + step * 7_919 + stream * 104_729 + seed * 503
    )
    return (values.float() + 0.5) / (MASK + 1.0)


def uniform_for_items(user_ids, item_ids, step, stream, seed):
    values = _mix(
        user_ids[:, None].long() * 1_103_515_245
        + item_ids.long() * 48_271
        + step * 7_919
        + stream * 104_729
        + seed * 503
    )
    return (values.float() + 0.5) / (MASK + 1.0)


def normal(user_ids, step, stream, seed, width=None):
    first = uniform(user_ids, step, stream, seed, width).clamp_min(1e-7)
    second = uniform(user_ids, step, stream + 1, seed, width)
    return torch.sqrt(-2.0 * torch.log(first)) * torch.cos(
        2.0 * torch.pi * second
    )
