"""Fixed hidden neural equations for repeated-creator Feed supply."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch


@dataclass(frozen=True)
class NeuralFeedSupplyTeacher:
    input_weight: torch.Tensor
    input_bias: torch.Tensor
    cross_weight: torch.Tensor
    head_weight: torch.Tensor
    head_bias: torch.Tensor

    def __call__(self, inputs):
        hidden = torch.nn.functional.silu(
            inputs @ self.input_weight + self.input_bias
        )
        crossed = inputs * torch.roll(inputs, shifts=5, dims=-1)
        hidden = hidden + 0.35 * torch.sin(crossed @ self.cross_weight)
        return hidden @ self.head_weight + self.head_bias


def build_neural_feed_supply_teacher(device, input_dim=15, hidden_dim=64):
    generator = torch.Generator(device=device).manual_seed(2026082417)

    def normal(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator, device=device) * scale

    return NeuralFeedSupplyTeacher(
        normal(input_dim, hidden_dim, scale=1.0 / sqrt(input_dim)),
        normal(hidden_dim, scale=0.15),
        normal(input_dim, hidden_dim, scale=1.0 / sqrt(input_dim)),
        normal(hidden_dim, 6, scale=1.10 / sqrt(hidden_dim)),
        torch.tensor([0.0, -1.40, -1.00, 0.0, -4.20, -0.15], device=device),
    )
