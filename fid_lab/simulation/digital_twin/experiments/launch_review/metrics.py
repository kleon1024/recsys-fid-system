"""Clustered user-level estimators and the preregistered Feed gate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ...contracts import (
    AppEventBatch,
    EventType,
    PublishFailureReason,
    Surface,
)


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

CROSS_REQUEST_METRICS = {
    "posting_entry": (EventType.SURFACE_ENTRY, Surface.POSTING),
    "create": (EventType.CREATE, Surface.POSTING),
    "publish": (EventType.PUBLISH, Surface.POSTING),
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


def _cross_request_user_metric(
    events: AppEventBatch,
    users: int,
    event_type: EventType,
    surface: Surface,
) -> torch.Tensor:
    selected = (
        (events.user_id >= 0)
        & (events.surface == int(surface))
        & events.event(event_type)
    )
    result = torch.zeros(users, device=events.user_id.device)
    result.scatter_add_(
        0,
        events.user_id[selected],
        torch.ones(int(selected.sum()), device=events.user_id.device),
    )
    return result


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


@dataclass
class StreamingExperimentMetrics:
    """Bounded-memory user-clustered metrics for long or large A/B windows."""

    users: int
    device: torch.device | str
    control_cell: int = 0
    treatment_cell: int = 1
    _cell_by_user: torch.Tensor = field(init=False)
    _metrics: dict[str, torch.Tensor] = field(init=False)
    _cross_request: dict[str, torch.Tensor] = field(init=False)
    _funnel_counts: dict[str, int] = field(init=False)
    _publish_failures: dict[str, int] = field(init=False)
    _cohort_frozen: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.users <= 0:
            raise ValueError("streaming experiment users must be positive")
        target = torch.device(self.device)
        self._cell_by_user = torch.full(
            (self.users,), -1, dtype=torch.long, device=target,
        )
        metric_names = ("dwell_seconds", *COUNT_METRICS)
        self._metrics = {
            name: torch.zeros(
                (2, self.users), dtype=torch.float32, device=target,
            )
            for name in metric_names
        }
        self._cross_request = {
            name: torch.zeros(
                self.users, dtype=torch.float32, device=target,
            )
            for name in CROSS_REQUEST_METRICS
        }
        self._funnel_counts = {
            "posting_entry": 0,
            "create": 0,
            "publish": 0,
            "publish_failed": 0,
        }
        self._publish_failures = {
            reason.name.lower(): 0 for reason in PublishFailureReason
        }

    def append(
        self,
        events: AppEventBatch,
        *,
        cross_request_only: bool = False,
    ) -> None:
        if not len(events.event_id):
            return
        self._append_funnel_diagnostics(events)
        impression = torch.zeros_like(events.event_type, dtype=torch.bool)
        if not self._cohort_frozen:
            impression = events.event(EventType.IMPRESSION) & torch.isin(
                events.experiment_cell,
                torch.tensor(
                    [self.control_cell, self.treatment_cell],
                    device=events.event_id.device,
                ),
            )
        if impression.any():
            user = events.user_id[impression]
            incoming = events.experiment_cell[impression]
            prior = self._cell_by_user[user]
            if ((prior >= 0) & (prior != incoming)).any():
                raise ValueError("user changed experiment cell during A/B")
            self._cell_by_user[user] = incoming
        if not cross_request_only:
            self._append_feed_metrics(events)
        for name, (event_type, surface) in CROSS_REQUEST_METRICS.items():
            selected = (
                (events.user_id >= 0)
                & (events.surface == int(surface))
                & events.event(event_type)
            )
            selected &= self._cell_by_user[events.user_id.clamp_min(0)] >= 0
            if selected.any():
                self._cross_request[name].scatter_add_(
                    0,
                    events.user_id[selected],
                    torch.ones(int(selected.sum()), device=events.event_id.device),
                )

    def freeze_cohort(self) -> dict[str, int]:
        self._cohort_frozen = True
        return self.sample()

    def sample(self) -> dict[str, int]:
        return {
            "control_triggered_users": int(
                (self._cell_by_user == self.control_cell).sum()
            ),
            "treatment_triggered_users": int(
                (self._cell_by_user == self.treatment_cell).sum()
            ),
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "funnel_event_counts": dict(self._funnel_counts),
            "publish_failure_reasons": dict(self._publish_failures),
        }

    def _append_feed_metrics(self, events: AppEventBatch) -> None:
        for row, cell in enumerate((self.control_cell, self.treatment_cell)):
            for name, event_type in {
                "dwell_seconds": EventType.DWELL,
                **COUNT_METRICS,
            }.items():
                selected = (
                    (events.experiment_cell == cell)
                    & (events.user_id >= 0)
                    & events.event(event_type)
                )
                if not selected.any():
                    continue
                value = (
                    events.duration_ms[selected].float() / 1_000.0
                    if event_type is EventType.DWELL
                    else torch.ones(
                        int(selected.sum()), device=events.event_id.device,
                    )
                )
                self._metrics[name][row].scatter_add_(
                    0, events.user_id[selected], value,
                )

    def _append_funnel_diagnostics(self, events: AppEventBatch) -> None:
        event_specs = {
            "posting_entry": (EventType.SURFACE_ENTRY, Surface.POSTING),
            "create": (EventType.CREATE, Surface.POSTING),
            "publish": (EventType.PUBLISH, Surface.POSTING),
            "publish_failed": (EventType.PUBLISH_FAILED, Surface.POSTING),
        }
        for name, (event_type, surface) in event_specs.items():
            selected = events.event(event_type) & (
                events.surface == int(surface)
            )
            self._funnel_counts[name] += int(selected.sum())
        failed = events.event(EventType.PUBLISH_FAILED)
        for reason in PublishFailureReason:
            self._publish_failures[reason.name.lower()] += int((
                failed & (events.value == int(reason))
            ).sum())

    def analyze(self) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
        control = self._cell_by_user == self.control_cell
        treatment = self._cell_by_user == self.treatment_cell
        metrics = {
            name: _estimate(value[0][control], value[1][treatment])
            for name, value in self._metrics.items()
        }
        metrics.update({
            name: _estimate(value[control], value[treatment])
            for name, value in self._cross_request.items()
        })
        return metrics, self.sample()


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
    for name, (event_type, surface) in CROSS_REQUEST_METRICS.items():
        values = _cross_request_user_metric(
            events, users, event_type, surface,
        )
        metrics[name] = _estimate(
            values[control_users], values[treatment_users],
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
