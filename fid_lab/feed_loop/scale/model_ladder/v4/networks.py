"""Candidate-conditioned Feed rankers for the V4 request contract."""

from __future__ import annotations

import torch
from torch import nn

from ....world_model.benchmark.neural import (
    DINRequestRanker,
    SlateTransformerRanker,
)


class SingleTaskDIN(DINRequestRanker):
    def forward(self, candidates, sequence):
        return super().forward(candidates, sequence).unsqueeze(2)


class SingleTaskTransformer(SlateTransformerRanker):
    def forward(self, candidates, sequence):
        return super().forward(candidates, sequence).unsqueeze(2)

class MMoERanker(DINRequestRanker):
    def __init__(self, tasks: int, feature_dim=28, sequence_dim=8, width=64,
                 experts=6) -> None:
        super().__init__(feature_dim, sequence_dim, width)
        self.task_count = tasks
        representation_dim = width * 3
        self.head = nn.Identity()
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(representation_dim, 128), nn.SiLU(),
                nn.Linear(128, width), nn.SiLU(),
            )
            for _ in range(experts)
        )
        self.gates = nn.ModuleList(
            nn.Linear(representation_dim, experts) for _ in range(tasks)
        )
        self.towers = nn.ModuleList(nn.Linear(width, 1) for _ in range(tasks))

    def forward(self, candidates, sequence):
        representation = self.representation(candidates, sequence)
        expert_values = torch.stack(
            tuple(expert(representation) for expert in self.experts), dim=-2
        )
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(representation), dim=-1)
            mixture = (expert_values * weights[..., None]).sum(dim=-2)
            outputs.append(tower(mixture))
        return torch.cat(outputs, dim=-1)


class PLERanker(DINRequestRanker):
    """One-level PLE: shared plus task-specific experts and task gates."""

    def __init__(self, tasks: int, feature_dim=28, sequence_dim=8, width=64,
                 shared_experts=3, task_experts=2) -> None:
        super().__init__(feature_dim, sequence_dim, width)
        self.task_count = tasks
        representation_dim = width * 3
        self.head = nn.Identity()
        self.shared = nn.ModuleList(
            _expert(representation_dim, width) for _ in range(shared_experts)
        )
        self.specific = nn.ModuleList(
            nn.ModuleList(
                _expert(representation_dim, width) for _ in range(task_experts)
            )
            for _ in range(tasks)
        )
        expert_count = shared_experts + task_experts
        self.gates = nn.ModuleList(
            nn.Linear(representation_dim, expert_count) for _ in range(tasks)
        )
        self.towers = nn.ModuleList(nn.Linear(width, 1) for _ in range(tasks))

    def forward(self, candidates, sequence):
        representation = self.representation(candidates, sequence)
        shared = tuple(expert(representation) for expert in self.shared)
        outputs = []
        for specific, gate, tower in zip(self.specific, self.gates, self.towers):
            values = torch.stack(
                (*shared, *(expert(representation) for expert in specific)), dim=-2
            )
            weights = torch.softmax(gate(representation), dim=-1)
            outputs.append(tower((values * weights[..., None]).sum(dim=-2)))
        return torch.cat(outputs, dim=-1)


def _expert(inputs: int, width: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(inputs, 128), nn.SiLU(), nn.Linear(128, width), nn.SiLU()
    )
