"""Stable serving boundary between policy constraints and learned scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..contracts import TwinPolicy


class CandidateScorer(Protocol):
    model_id: str
    architecture: str

    def score(
        self, features: torch.Tensor, surface: torch.Tensor,
    ) -> torch.Tensor: ...

    def manifest(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ServingStack:
    strategy: TwinPolicy
    coarse_model: CandidateScorer | None = None
    fine_model: CandidateScorer | None = None
    coarse_model_weight: float = 1.0
    fine_model_weight: float = 1.0

    def __post_init__(self):
        for value in (self.coarse_model_weight, self.fine_model_weight):
            if not 0.0 <= value <= 1.0:
                raise ValueError("learned model blend weights must be in [0, 1]")

    @property
    def name(self) -> str:
        versions = [self.strategy.name]
        if self.coarse_model is not None:
            versions.append(f"coarse={self.coarse_model.model_id}")
        if self.fine_model is not None:
            versions.append(f"fine={self.fine_model.model_id}")
        return "+".join(versions)

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "strategy": self.strategy.manifest(),
            "coarse_model": (
                self.coarse_model.manifest() if self.coarse_model else None
            ),
            "fine_model": (
                self.fine_model.manifest() if self.fine_model else None
            ),
            "coarse_model_weight": self.coarse_model_weight,
            "fine_model_weight": self.fine_model_weight,
        }


def as_serving_stack(value: TwinPolicy | ServingStack) -> ServingStack:
    if isinstance(value, ServingStack):
        return value
    return ServingStack(strategy=value)
