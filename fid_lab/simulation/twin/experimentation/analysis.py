"""Population-slice and experiment-induced sample diagnostics."""

from __future__ import annotations

import torch

from ..metrics import METRIC_INDEX
from .mixed import MixedTwinRun


def _concat_user_field(run: MixedTwinRun, name: str):
    return torch.cat(tuple(
        getattr(batch, name) for batch in run.snapshot.users
    ))


def _cell_summary(metrics, registered, active, mask):
    count = int(mask.sum())
    if count == 0:
        return {
            "users": 0, "registered_rate": 0.0, "active_rate": 0.0,
            "requests_per_user": 0.0, "stay_seconds_per_user": 0.0,
            "synthetic_lt_per_user": 0.0,
        }
    selected = metrics[mask]
    return {
        "users": count,
        "registered_rate": float(registered[mask].float().mean()),
        "active_rate": float(active[mask].float().mean()),
        "requests_per_user": float(
            selected[:, METRIC_INDEX["requests"]].mean()
        ),
        "stay_seconds_per_user": float(
            selected[:, METRIC_INDEX["stay_seconds"]].mean()
        ),
        "synthetic_lt_per_user": float(
            selected[:, METRIC_INDEX["synthetic_lt_measurement"]].mean()
        ),
    }

def mixed_sample_report(run: MixedTwinRun):
    metrics = torch.cat(run.user_metrics)
    assigned = torch.cat(run.assigned_treatment)
    registered = _concat_user_field(run, "registered")
    active = _concat_user_field(run, "active")
    report = {
        name: _cell_summary(
            metrics, registered, active,
            assigned if name == "treatment" else ~assigned,
        )
        for name in ("control", "treatment")
    }
    dimensions = {
        "country": _concat_user_field(run, "country"),
        "lifecycle": _concat_user_field(run, "lifecycle"),
        "activity_tier": _concat_user_field(run, "activity_tier"),
        "socioeconomic": _concat_user_field(run, "socioeconomic"),
        "acquisition_channel": _concat_user_field(run, "acquisition_channel"),
        "timezone_offset": _concat_user_field(run, "timezone_offset"),
        "cold_start": (
            _concat_user_field(run, "cold_start_confidence") < 0.25
        ).long(),
    }
    slices = {}
    for dimension, values in dimensions.items():
        slices[dimension] = {}
        for value in torch.unique(values).tolist():
            segment = values == value
            slices[dimension][str(value)] = {
                cell: _cell_summary(
                    metrics, registered, active,
                    segment & (assigned if cell == "treatment" else ~assigned),
                )
                for cell in ("control", "treatment")
            }
    return {
        "experiment_cells": report,
        "slices": slices,
        "interpretation": (
            "Requests, activity, and registered samples are post-treatment "
            "outcomes generated inside the shared experiment world."
        ),
    }
