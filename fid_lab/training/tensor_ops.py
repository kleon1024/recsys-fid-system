"""Shared request-level tensor operations used by ranking trainers."""

from __future__ import annotations


def gather_candidates(values, indices):
    """Gather candidate rows while preserving any trailing feature dimension."""
    if values.ndim == 3:
        expanded = indices[:, :, None].expand(-1, -1, values.shape[2])
        return values.gather(1, expanded)
    return values.gather(1, indices)
