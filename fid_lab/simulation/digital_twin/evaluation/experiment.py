"""Factual user-clustered A/B diagnostics over one evolving world."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from scipy.stats import chisquare
import torch

from ..contracts import AppEventBatch, EventType, Surface
from ..samples.contracts import RequestCandidateTrace


def _assigned_users(
    trace: RequestCandidateTrace | tuple[RequestCandidateTrace, ...],
    cell: int,
    surface: Surface,
) -> torch.Tensor:
    traces = trace if isinstance(trace, tuple) else (trace,)
    selected_users = tuple(
        value.user_id[
            (value.experiment_cell == cell)
            & (value.surface == int(surface))
        ]
        for value in traces
    )
    return torch.unique(torch.cat(selected_users), sorted=True)


def _user_metric(
    events: AppEventBatch,
    users: torch.Tensor,
    cell: int,
    event_type: EventType,
) -> np.ndarray:
    if not len(users):
        return np.empty(0, dtype=np.float64)
    selected = (
        (events.experiment_cell == cell)
        & events.event(event_type)
        & torch.isin(events.user_id, users)
    )
    value = torch.zeros(len(users), device=events.user_id.device)
    location = torch.searchsorted(users, events.user_id[selected])
    increment = (
        events.duration_ms[selected].float() / 1_000.0
        if event_type == EventType.DWELL
        else torch.ones(int(selected.sum()), device=events.user_id.device)
    )
    value.scatter_add_(0, location, increment)
    return value.double().cpu().numpy()


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
    control = _assigned_users(trace, 0, surface)
    treatment = _assigned_users(trace, 1, surface)
    contamination = torch.isin(control, treatment)
    observed = np.asarray([len(control), len(treatment)], dtype=float)
    total_fraction = control_fraction + treatment_fraction
    expected = observed.sum() * np.asarray([
        control_fraction / total_fraction,
        treatment_fraction / total_fraction,
    ])
    srm_p = (
        None if observed.sum() == 0 else float(chisquare(observed, expected).pvalue)
    )
    metrics = {}
    for name, event_type in {
        "dwell_seconds": EventType.DWELL,
        "play_3s": EventType.PLAY_3S,
        "long_view": EventType.LONG_VIEW,
        "complete": EventType.COMPLETE,
        "like": EventType.LIKE,
        "share": EventType.SHARE,
        "negative": EventType.NEGATIVE,
        "session_end": EventType.SESSION_END,
    }.items():
        metrics[name] = _effect(
            _user_metric(events, control, 0, event_type),
            _user_metric(events, treatment, 1, event_type),
        )
    return {
        "schema": "factual-user-ab-evaluation/v1",
        "surface": surface.name.lower(),
        "assignment_unit": "user",
        "control_users": len(control),
        "treatment_users": len(treatment),
        "cross_cell_users": int(contamination.sum()),
        "srm_p_value": srm_p,
        "metrics": metrics,
    }


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
