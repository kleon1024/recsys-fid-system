"""Versioned online parameter server with idempotency and staleness control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import TASKS


@dataclass(frozen=True, eq=False)
class ParameterSnapshot:
    version: int
    weights: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class UpdateResult:
    version: int
    applied: bool
    reason: str


class VersionedParameterServer:
    def __init__(
        self,
        feature_dim: int = 256,
        tasks: tuple[str, ...] = TASKS,
        max_staleness: int = 2,
    ) -> None:
        self.feature_dim = feature_dim
        self.tasks = tasks
        self.max_staleness = max_staleness
        self._weights = np.zeros((len(tasks), feature_dim), dtype=np.float64)
        self._bias = np.zeros(len(tasks), dtype=np.float64)
        self._version = 0
        self._applied_updates: set[str] = set()

    def snapshot(self) -> ParameterSnapshot:
        return ParameterSnapshot(self._version, self._weights.copy(), self._bias.copy())

    def apply(
        self,
        update_id: str,
        base_version: int,
        weight_gradient: np.ndarray,
        bias_gradient: np.ndarray,
        learning_rate: float,
    ) -> UpdateResult:
        if update_id in self._applied_updates:
            return UpdateResult(self._version, False, "duplicate_update")
        if self._version - base_version > self.max_staleness:
            return UpdateResult(self._version, False, "stale_gradient")
        if weight_gradient.shape != self._weights.shape or bias_gradient.shape != self._bias.shape:
            raise ValueError("gradient shape does not match parameter shape")
        if not np.isfinite(weight_gradient).all() or not np.isfinite(bias_gradient).all():
            raise ValueError("gradients must be finite")
        self._weights -= learning_rate * weight_gradient
        self._bias -= learning_rate * bias_gradient
        self._version += 1
        self._applied_updates.add(update_id)
        return UpdateResult(self._version, True, "applied")
