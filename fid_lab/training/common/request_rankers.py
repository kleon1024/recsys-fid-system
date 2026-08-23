"""Shared request-shaped multi-task ranking architectures."""

from __future__ import annotations

from torch import nn


class RequestLinearRanker(nn.Module):
    def __init__(self, width, tasks):
        super().__init__()
        self.heads = nn.ModuleDict({task: nn.Linear(width, 1) for task in tasks})

    def forward(self, features, candidate_semantic, history):
        del candidate_semantic, history
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        return {
            task: head(flat).reshape(shape) for task, head in self.heads.items()
        }
