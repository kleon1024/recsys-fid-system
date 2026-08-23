"""Single authority for the frozen V2/V3 nonlinear stay oracle."""

from __future__ import annotations

import torch


def nonlinear_stay_adjustment(features: torch.Tensor) -> torch.Tensor:
    affinity = features[..., 0]
    quality = features[..., 1]
    short_match = features[..., 4]
    long_match = features[..., 11]
    user_segment = torch.remainder(torch.round(features[..., 14] * 1023), 7)
    item_segment = torch.remainder(torch.round(features[..., 15] * 4095), 7)
    segment_match = 1.0 - torch.abs(user_segment - item_segment) / 6.0
    threshold = ((affinity > 0.28) & (quality > 0.62)).float()
    sequence_novelty = short_match * (1.0 - long_match)
    periodic = torch.sin(torch.pi * features[..., 17] * (1.0 + features[..., 10]))
    return (
        0.75 * torch.tanh(2.4 * affinity * quality)
        + 0.50 * threshold
        + 0.35 * segment_match
        + 0.45 * sequence_novelty
        + 0.25 * periodic
        - 1.45
    )
