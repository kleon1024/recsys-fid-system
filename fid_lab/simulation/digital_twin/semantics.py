"""Versioned semantic signal contract shared by the world and platform catalog."""

from __future__ import annotations

import torch


SEMANTIC_SIGNAL_VERSION = "observable-hidden-semantic-v3"

CONTENT_TOPIC_RESIDUAL_WEIGHT = 0.65
HIDDEN_ITEM_RESIDUAL_WEIGHT = 0.42
USER_LONG_RESIDUAL_WEIGHT = 0.55
USER_SHORT_RESIDUAL_WEIGHT = 0.48


def mix_direction(
    signal: torch.Tensor,
    residual: torch.Tensor,
    residual_weight: float,
) -> torch.Tensor:
    """Mix directions without letting vector width amplify private noise."""
    if signal.shape != residual.shape:
        raise ValueError("semantic signal and residual shapes must match")
    if residual_weight < 0.0:
        raise ValueError("semantic residual weight must be nonnegative")
    unit_signal = torch.nn.functional.normalize(signal, dim=-1)
    unit_residual = torch.nn.functional.normalize(residual, dim=-1)
    return torch.nn.functional.normalize(
        unit_signal + residual_weight * unit_residual,
        dim=-1,
    )
