"""Clustered user-level estimators and the preregistered Feed gate."""

from __future__ import annotations

import math

import torch

from ...contracts import AppEventBatch, EventType


COUNT_METRICS = {
    "play_3s": EventType.PLAY_3S,
    "long_view": EventType.LONG_VIEW,
    "complete": EventType.COMPLETE,
    "like": EventType.LIKE,
    "comment": EventType.COMMENT,
    "share": EventType.SHARE,
    "follow": EventType.FOLLOW,
    "negative": EventType.NEGATIVE,
    "session_end": EventType.SESSION_END,
}


def _user_metric(
    events: AppEventBatch,
    cell: int,
    users: int,
    event_type: EventType,
) -> torch.Tensor:
    selected = (
        (events.experiment_cell == cell)
        & (events.user_id >= 0)
        & events.event(event_type)
    )
    result = torch.zeros(users, device=events.user_id.device)
    if event_type is EventType.DWELL:
        result.scatter_add_(
            0,
            events.user_id[selected],
            events.duration_ms[selected].float() / 1_000.0,
        )
    else:
        result.scatter_add_(
            0,
            events.user_id[selected],
            torch.ones(int(selected.sum()), device=events.user_id.device),
        )
    return result


def _cell_users(events: AppEventBatch, cell: int, users: int) -> torch.Tensor:
    impression = (events.experiment_cell == cell) & events.event(
        EventType.IMPRESSION
    )
    present = torch.zeros(users, device=events.user_id.device, dtype=torch.bool)
    present[events.user_id[impression]] = True
    return present


def _estimate(
    control: torch.Tensor,
    treatment: torch.Tensor,
) -> dict[str, float]:
    if len(control) < 2 or len(treatment) < 2:
        return {
            name: float("nan")
            for name in (
                "control_mean", "treatment_mean", "absolute_delta",
                "relative_delta", "ci95_low", "ci95_high",
                "standard_error", "mde80_absolute", "mde80_relative",
            )
        }
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    delta = treatment_mean - control_mean
    standard_error = torch.sqrt(
        control.var(unbiased=True) / len(control)
        + treatment.var(unbiased=True) / len(treatment)
    )
    mde80 = 2.80 * standard_error
    return {
        "control_mean": float(control_mean),
        "treatment_mean": float(treatment_mean),
        "absolute_delta": float(delta),
        "relative_delta": float(delta / control_mean.clamp_min(1e-12)),
        "ci95_low": float(delta - 1.96 * standard_error),
        "ci95_high": float(delta + 1.96 * standard_error),
        "standard_error": float(standard_error),
        "mde80_absolute": float(mde80),
        "mde80_relative": float(
            mde80 / control_mean.abs().clamp_min(1e-12)
        ),
    }


def analyze_experiment(
    events: AppEventBatch,
    users: int,
    control_cell: int = 0,
    treatment_cell: int = 1,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    control_users = _cell_users(events, control_cell, users)
    treatment_users = _cell_users(events, treatment_cell, users)
    metrics = {}
    for name, event_type in {
        "dwell_seconds": EventType.DWELL,
        **COUNT_METRICS,
    }.items():
        metrics[name] = _estimate(
            _user_metric(events, control_cell, users, event_type)[control_users],
            _user_metric(
                events, treatment_cell, users, event_type,
            )[treatment_users],
        )
    return metrics, {
        "control_triggered_users": int(control_users.sum()),
        "treatment_triggered_users": int(treatment_users.sum()),
    }


def validate_aa(
    metrics: dict[str, dict[str, float]],
) -> tuple[bool, str]:
    for name in ("dwell_seconds", "negative"):
        metric = metrics[name]
        values = tuple(metric.values())
        if not all(math.isfinite(value) for value in values):
            return False, f"A/A {name} contains a non-finite metric"
        if not metric["ci95_low"] <= 0.0 <= metric["ci95_high"]:
            return False, f"A/A {name} confidence interval excludes zero"
    return True, "A/A primary and guardrail intervals include zero"


def decide_launch(
    metrics: dict[str, dict[str, float]],
    sample: dict[str, int],
    minimum_triggered_users: int,
) -> tuple[str, str]:
    if min(sample.values()) < minimum_triggered_users:
        return "hold", "triggered-user sample is below the preregistered gate"
    dwell = metrics["dwell_seconds"]
    negative = metrics["negative"]
    if not all(
        math.isfinite(value)
        for metric in metrics.values()
        for value in metric.values()
    ):
        return "hold", "non-finite experiment metric"
    if dwell["ci95_high"] < 0.0:
        return "reject", "stay significantly decreases"
    if negative["ci95_low"] > 0.0:
        return "reject", "negative feedback significantly increases"
    if dwell["ci95_low"] <= 0.0:
        return "hold", "stay confidence interval crosses zero"
    return "promote", "stay improves and negative-feedback guardrail passes"
