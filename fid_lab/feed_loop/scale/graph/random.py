"""Counter-based device RNG invariant to batch partitioning."""

from __future__ import annotations

import torch


MODULUS = 2_147_483_647


def uniform(user_ids, step, stream, seed, width=None):
    keys = user_ids.long()
    if width is not None:
        position = torch.arange(width, device=user_ids.device, dtype=torch.long)
        keys = keys[:, None] * 1_103_515_245 + position[None, :] * 12_345
    values = torch.remainder(
        keys * 48_271 + step * 7_919 + stream * 104_729 + seed * 503,
        MODULUS,
    )
    return (values.float() + 0.5) / MODULUS


def uniform_for_items(user_ids, item_ids, step, stream, seed):
    values = torch.remainder(
        user_ids[:, None].long() * 1_103_515_245
        + item_ids.long() * 48_271
        + step * 7_919
        + stream * 104_729
        + seed * 503,
        MODULUS,
    )
    return (values.float() + 0.5) / MODULUS


def normal(user_ids, step, stream, seed, width=None):
    first = uniform(user_ids, step, stream, seed, width).clamp_min(1e-7)
    second = uniform(user_ids, step, stream + 1, seed, width)
    return torch.sqrt(-2.0 * torch.log(first)) * torch.cos(
        2.0 * torch.pi * second
    )
