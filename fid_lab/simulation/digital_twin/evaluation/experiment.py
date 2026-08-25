"""Factual user-clustered A/B diagnostics over one evolving world."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from scipy.stats import chisquare
import torch

from ..contracts import AppEventBatch, EventType, Surface
from ..samples.contracts import RequestCandidateTrace


AB_EVENT_TYPES = {
    "dwell_seconds": EventType.DWELL,
    "play_3s": EventType.PLAY_3S,
    "long_view": EventType.LONG_VIEW,
    "complete": EventType.COMPLETE,
    "like": EventType.LIKE,
    "share": EventType.SHARE,
    "negative": EventType.NEGATIVE,
    "session_end": EventType.SESSION_END,
}


class FactualABAccumulator:
    """Bounded user-clustered metrics for a multi-tick factual experiment."""

    def __init__(
        self,
        users: int,
        *,
        control_fraction: float,
        treatment_fraction: float,
        surface: Surface = Surface.FEED,
        device: torch.device | str = "cpu",
    ) -> None:
        if users <= 0:
            raise ValueError("A/B accumulator requires a positive user universe")
        self.users = users
        self.control_fraction = control_fraction
        self.treatment_fraction = treatment_fraction
        self.surface = surface
        self.assignment = torch.zeros((2, users), dtype=torch.bool, device=device)
        self.values = torch.zeros(
            (2, len(AB_EVENT_TYPES), users), dtype=torch.float64, device=device,
        )

    def update(
        self,
        trace: RequestCandidateTrace,
        events: AppEventBatch,
    ) -> None:
        for cell in (0, 1):
            assigned = (
                (trace.experiment_cell == cell)
                & (trace.surface == int(self.surface))
            )
            user = trace.user_id[assigned]
            if len(user):
                self.assignment[cell, torch.unique(user)] = True
            for metric, event_type in enumerate(AB_EVENT_TYPES.values()):
                selected = (
                    (events.experiment_cell == cell)
                    & (events.surface == int(self.surface))
                    & events.event(event_type)
                    & (events.user_id >= 0)
                    & (events.user_id < self.users)
                )
                event_user = events.user_id[selected]
                if not len(event_user):
                    continue
                increment = (
                    events.duration_ms[selected].double() / 1_000.0
                    if event_type == EventType.DWELL
                    else torch.ones(
                        len(event_user), dtype=torch.float64,
                        device=event_user.device,
                    )
                )
                self.values[cell, metric].index_add_(
                    0, event_user, increment,
                )

    def report(self) -> dict[str, object]:
        control = self.assignment[0]
        treatment = self.assignment[1]
        observed = np.asarray([
            int(control.sum()), int(treatment.sum()),
        ], dtype=float)
        total_fraction = self.control_fraction + self.treatment_fraction
        expected = observed.sum() * np.asarray([
            self.control_fraction / total_fraction,
            self.treatment_fraction / total_fraction,
        ])
        srm_p = (
            None
            if observed.sum() == 0
            else float(chisquare(observed, expected).pvalue)
        )
        metrics = {}
        for index, name in enumerate(AB_EVENT_TYPES):
            metrics[name] = _effect(
                self.values[0, index, control].cpu().numpy(),
                self.values[1, index, treatment].cpu().numpy(),
            )
        return {
            "schema": "factual-user-ab-evaluation/v1",
            "surface": self.surface.name.lower(),
            "assignment_unit": "user",
            "control_users": int(control.sum()),
            "treatment_users": int(treatment.sum()),
            "cross_cell_users": int((control & treatment).sum()),
            "srm_p_value": srm_p,
            "metrics": metrics,
        }


def _effect(control: np.ndarray, treatment: np.ndarray) -> dict[str, object]:
    if len(control) < 2 or len(treatment) < 2:
        return {"status": "insufficient_users"}
    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    delta = treatment_mean - control_mean
    standard_error = float(np.sqrt(
        control.var(ddof=1) / len(control)
        + treatment.var(ddof=1) / len(treatment)
    ))
    z = delta / max(standard_error, 1e-12)
    return {
        "status": "estimated",
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "absolute_delta": delta,
        "relative_delta": (
            None if abs(control_mean) < 1e-12 else delta / abs(control_mean)
        ),
        "standard_error": standard_error,
        "ci95_low": delta - 1.96 * standard_error,
        "ci95_high": delta + 1.96 * standard_error,
        "p_value": erfc(abs(z) / sqrt(2.0)),
        "approximate_mde_80pct": 2.8 * standard_error,
    }


def factual_ab_report(
    trace: RequestCandidateTrace | tuple[RequestCandidateTrace, ...],
    events: AppEventBatch,
    *,
    control_fraction: float,
    treatment_fraction: float,
    surface: Surface = Surface.FEED,
) -> dict[str, object]:
    traces = trace if isinstance(trace, tuple) else (trace,)
    maximum = max(
        int(value.user_id.max()) if len(value.user_id) else -1
        for value in traces
    )
    if len(events.user_id):
        maximum = max(maximum, int(events.user_id.max()))
    accumulator = FactualABAccumulator(
        maximum + 1,
        control_fraction=control_fraction,
        treatment_fraction=treatment_fraction,
        surface=surface,
        device=events.user_id.device,
    )
    for value in traces:
        accumulator.update(value, events)
        events = AppEventBatch.empty(events.user_id.device)
    return accumulator.report()


def aa_decision(
    report: dict[str, object],
    *,
    primary_metric: str = "dwell_seconds",
    minimum_srm_p: float = 0.001,
) -> dict[str, object]:
    primary = report["metrics"][primary_metric]
    passes = (
        report["cross_cell_users"] == 0
        and report["srm_p_value"] is not None
        and report["srm_p_value"] >= minimum_srm_p
        and primary.get("status") == "estimated"
        and primary["ci95_low"] <= 0.0 <= primary["ci95_high"]
    )
    return {
        "decision": "pass" if passes else "hold",
        "primary_metric": primary_metric,
        "minimum_srm_p": minimum_srm_p,
        "reason": (
            "A/A assignment and zero-effect interval pass"
            if passes else "A/A assignment or zero-effect interval is not valid"
        ),
    }
