"""Primitive task and ranking-value contracts for V4 Feed training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_index: int
    kind: str
    loss_weight: float
    audit_weight: float
    scale: float = 1.0


TASKS = (
    TaskSpec("play_3s", 1, "binary", 0.35, 0.20),
    TaskSpec("stay", 2, "regression", 1.00, 0.40, 180.0),
    TaskSpec("completion", 3, "regression", 0.30, 0.00),
    TaskSpec("long_view", 5, "binary", 0.45, 0.30),
    TaskSpec("quality_long_view", 6, "binary", 0.55, 0.00),
    TaskSpec("like", 7, "binary", 0.20, 0.10),
    TaskSpec("negative_feedback", 8, "binary", 0.25, -0.10),
    TaskSpec("anchor_click", 9, "binary", 0.10, 0.00),
    TaskSpec("conversion", 12, "binary", 0.05, 0.00),
    TaskSpec("returned_next_session", 13, "binary", 0.15, 0.00),
)

LONG_VIEW_TASK = next(
    index for index, task in enumerate(TASKS) if task.name == "long_view"
)
