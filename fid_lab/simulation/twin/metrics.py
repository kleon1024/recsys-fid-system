"""User-level mature outcome metrics shared by all experiment arms."""

from __future__ import annotations

import torch

from .contracts import Surface
from .exchange import ObservableResponse, TASKS


METRICS = (
    "requests", "stay_seconds", *TASKS,
    *(f"surface_requests:{surface.name.lower()}" for surface in Surface),
    "active_days", "synthetic_lt_measurement",
)
METRIC_INDEX = {name: index for index, name in enumerate(METRICS)}


def empty_user_metrics(users: int, device):
    return torch.zeros(users, len(METRICS), device=device)


def accumulate_metrics(
    metrics: torch.Tensor,
    response: ObservableResponse,
    surface: torch.Tensor,
) -> None:
    active = response.active.float()
    metrics[:, METRIC_INDEX["requests"]] += active
    metrics[:, METRIC_INDEX["stay_seconds"]] += response.stay_seconds
    for index, name in enumerate(TASKS):
        metrics[:, METRIC_INDEX[name]] += response.task[:, index].float()
    for value in Surface:
        metrics[:, METRIC_INDEX[f"surface_requests:{value.name.lower()}"]] += (
            (surface == int(value)) & response.active
        ).float()
    positive = (
        response.event("like").float()
        + 1.5 * response.event("share").float()
        + 1.8 * response.event("follow").float()
        + 2.0 * response.event("order").float()
        + 2.4 * response.event("payment").float()
        + 1.2 * response.event("publish").float()
    )
    commercialization = (
        response.event("payment").float()
        + 0.20 * response.event("click").float()
    )
    measurement = (
        response.stay_seconds / 60.0
        + 0.12 * positive
        + 0.05 * commercialization
        - 0.30 * response.event("negative").float()
    ).clamp_min(0.0)
    metrics[:, METRIC_INDEX["synthetic_lt_measurement"]] += measurement


def add_active_day(metrics: torch.Tensor, active: torch.Tensor):
    value = active.float()
    metrics[:, METRIC_INDEX["active_days"]] += value
    metrics[:, METRIC_INDEX["synthetic_lt_measurement"]] += 0.35 * value


def summarize_user_metrics(values: torch.Tensor) -> dict[str, float]:
    request = values[:, METRIC_INDEX["requests"]].sum().clamp_min(1.0)
    users = max(len(values), 1)
    report = {
        "users": users,
        "requests": int(request),
        "stay_seconds_per_user": float(
            values[:, METRIC_INDEX["stay_seconds"]].sum() / users
        ),
        "synthetic_lt_per_user": float(
            values[:, METRIC_INDEX["synthetic_lt_measurement"]].sum() / users
        ),
        "active_days_per_user": float(
            values[:, METRIC_INDEX["active_days"]].sum() / users
        ),
    }
    for name in TASKS:
        report[f"{name}_rate"] = float(
            values[:, METRIC_INDEX[name]].sum() / request
        )
    for surface in Surface:
        report[f"{surface.name.lower()}_request_share"] = float(
            values[:, METRIC_INDEX[f"surface_requests:{surface.name.lower()}"]].sum()
            / request
        )
    return report
