"""Reusable, inspectable Multi-gate Mixture-of-Experts implementation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MultiGateMixtureOfExperts(nn.Module):
    """Shared experts with one learned routing distribution per task."""

    def __init__(
        self,
        input_dim: int,
        tasks: Sequence[str],
        expert_count: int = 3,
        expert_dim: int = 32,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, expert_dim),
                nn.ReLU(),
            )
            for _ in range(expert_count)
        )
        self.gates = nn.ModuleDict(
            {task: nn.Linear(input_dim, expert_count) for task in self.tasks}
        )
        self.heads = nn.ModuleDict(
            {task: nn.Linear(expert_dim, 1) for task in self.tasks}
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_states = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        outputs: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            gate_weights = torch.softmax(self.gates[task](inputs), dim=1)
            task_state = (gate_weights[:, :, None] * expert_states).sum(dim=1)
            outputs[task] = self.heads[task](task_state).squeeze(1)
            outputs[f"gate:{task}"] = gate_weights
        return outputs
