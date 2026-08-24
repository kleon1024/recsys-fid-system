"""Bounded event-time trace window for delayed labels and streaming retraining."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..serving.trace import RequestTrace
from .contracts import TASK_MATURITY_STEPS


@dataclass
class TraceWindow:
    lookback_steps: int = 32
    trace: RequestTrace = field(default_factory=RequestTrace)

    def add(self, value: RequestTrace) -> None:
        self.trace.rows.extend(value.rows)

    def at(
        self,
        watermark_step: int,
        maximum_requests: int | None = None,
    ) -> RequestTrace:
        earliest = watermark_step - self.lookback_steps
        selected = [
            row for row in self.trace.rows
            if int(row["step"][0]) >= earliest
        ]
        selected.sort(key=lambda row: int(row["step"][0]))
        trace = RequestTrace(rows=selected)
        if maximum_requests is not None:
            return trace.sampled(maximum_requests, salt=watermark_step)
        return trace

    def prune(self, watermark_step: int) -> None:
        retain_from = watermark_step - self.lookback_steps
        self.trace.rows = [
            row for row in self.trace.rows
            if int(row["step"][0]) >= retain_from
        ]

    def manifest(self, watermark_step: int) -> dict[str, int]:
        selected = self.at(watermark_step)
        return {
            "stored_step_partitions": len(self.trace.rows),
            "training_step_partitions": len(selected.rows),
            "lookback_steps": self.lookback_steps,
            "max_label_maturity_steps": max(TASK_MATURITY_STEPS.values()),
        }
