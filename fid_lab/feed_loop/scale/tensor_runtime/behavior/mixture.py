"""Hidden cohort residuals layered over the externally trained response kernel."""

from __future__ import annotations

from math import sqrt

import torch


class HiddenResponseMixture:
    """Fixed neural structural residual; its inputs never enter serving features."""

    def __init__(self, device: torch.device, seed: int, inputs=14, width=48) -> None:
        generator = torch.Generator(device=device).manual_seed(seed + 91_337)

        def normal(*shape, scale=1.0):
            return torch.randn(
                *shape, generator=generator, device=device
            ) * scale

        self.input_weight = normal(inputs, width, scale=1.0 / sqrt(inputs))
        self.input_bias = normal(width, scale=0.12)
        self.expert_weight = normal(4, width, 8, scale=0.45 / sqrt(width))
        self.expert_bias = normal(4, 8, scale=0.04)

    def __call__(self, inputs: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(
            inputs @ self.input_weight + self.input_bias
        )
        crossed = hidden * torch.roll(hidden, 7, dims=1)
        hidden = hidden + 0.30 * torch.sin(crossed)
        expert = torch.einsum("bd,edt->bet", hidden, self.expert_weight)
        expert = expert + self.expert_bias[None]
        rows = torch.arange(len(inputs), device=inputs.device)
        return expert[rows, mixture]
