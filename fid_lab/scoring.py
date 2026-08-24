"""Shared score-composition primitives for request-level ranking."""

from __future__ import annotations


def request_standardize(values):
    """Remove per-request location and scale without crossing requests."""
    centered = values - values.mean(dim=1, keepdim=True)
    scale = values.std(dim=1, keepdim=True, correction=0).clamp_min(1e-4)
    return centered / scale
