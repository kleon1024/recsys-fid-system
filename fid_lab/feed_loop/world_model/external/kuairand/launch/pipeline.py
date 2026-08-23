"""Ordered, fail-closed launch review state for external world-model policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class LaunchStage(str, Enum):
    OFFLINE_CAPACITY = "offline_capacity"
    RANDOMIZED_CALIBRATION = "randomized_calibration"
    RANDOMIZED_OPE = "randomized_ope"
    STATEFUL_SHADOW = "stateful_shadow"
    SIMULATED_AB = "simulated_ab"
    REVIEW = "review"


STAGE_ORDER = tuple(LaunchStage)


@dataclass(frozen=True)
class LaunchState:
    active_authority: str = "v3"
    completed: tuple[LaunchStage, ...] = ()
    decision: str = "pending"

    def record(self, stage: LaunchStage, passed: bool) -> "LaunchState":
        if self.decision != "pending":
            raise ValueError("terminal launch state cannot advance")
        expected = STAGE_ORDER[len(self.completed)]
        if stage is not expected:
            raise ValueError(f"expected {expected.value}, received {stage.value}")
        if not passed:
            return replace(self, decision=f"hold_{stage.value}")
        completed = (*self.completed, stage)
        decision = "research_candidate_pass" if stage is LaunchStage.REVIEW else "pending"
        return replace(self, completed=completed, decision=decision)

    @property
    def can_promote_research_authority(self) -> bool:
        return self.decision == "research_candidate_pass" and self.completed == STAGE_ORDER
