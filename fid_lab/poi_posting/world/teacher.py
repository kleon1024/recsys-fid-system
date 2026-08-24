"""Fixed hidden neural structural equations for creator supply responses."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch


@dataclass(frozen=True)
class NeuralSupplyTeacher:
    input_weight: torch.Tensor
    input_bias: torch.Tensor
    cross_weight: torch.Tensor
    head_weight: torch.Tensor
    head_bias: torch.Tensor

    def __call__(self, inputs):
        hidden = torch.nn.functional.silu(
            inputs @ self.input_weight + self.input_bias
        )
        crossed = inputs * torch.roll(inputs, shifts=3, dims=-1)
        hidden = hidden + 0.35 * torch.sin(crossed @ self.cross_weight)
        return hidden @ self.head_weight + self.head_bias


def build_neural_supply_teacher(device, input_dim=12, hidden_dim=32):
    generator = torch.Generator(device=device).manual_seed(2026082407)
    def normal(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator, device=device) * scale
    return NeuralSupplyTeacher(
        normal(input_dim, hidden_dim, scale=1.0 / sqrt(input_dim)),
        normal(hidden_dim, scale=0.15),
        normal(input_dim, hidden_dim, scale=1.0 / sqrt(input_dim)),
        normal(hidden_dim, 6, scale=0.32 / sqrt(hidden_dim)),
        torch.tensor([0.0, -1.35, 0.15, 0.10, -4.10, -0.20], device=device),
    )
