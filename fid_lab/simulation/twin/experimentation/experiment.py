"""Randomized A/B over branches of one materialized pre-treatment world."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import torch

from ...experimentation.assignment import assign_binary_torch
from ..contracts import BASELINE_POLICY, TwinConfig, TwinPolicy
from ..kernel import DigitalTwinKernel, TwinRun
from ..metrics import METRIC_INDEX, METRICS, summarize_user_metrics
from .analysis import mixed_sample_report
from .mixed import MixedTwinRun, run_mixed_world_ab


@dataclass
class TwinExperiment:
    report: dict[str, object]
    preperiod: TwinRun
    control: TwinRun
    treatment: TwinRun
    mixed: MixedTwinRun


def _concatenate(runs):
    return torch.cat(tuple(value.double() for value in runs))


def _cuped_values(control_y, treatment_y, x, assigned):
    y = torch.where(assigned, treatment_y, control_y)
    x_centered = x - x.mean()
    variance = x_centered.square().mean()
    theta = (
        (x_centered * (y - y.mean())).mean() / variance
        if float(variance) > 1e-12 else torch.zeros((), device=y.device)
    )
    adjusted = y - theta * x_centered
    left, right = adjusted[~assigned], adjusted[assigned]
    difference = float(right.mean() - left.mean())
    standard_error = sqrt(
        float(left.var(unbiased=True) / len(left))
        + float(right.var(unbiased=True) / len(right))
    )
    return {
        "control_mean": float(control_y[~assigned].mean()),
        "treatment_mean": float(treatment_y[assigned].mean()),
        "difference": difference,
        "standard_error": standard_error,
        "confidence_interval": [
            difference - 1.96 * standard_error,
            difference + 1.96 * standard_error,
        ],
        "p_value": erfc(
            abs(difference / max(standard_error, 1e-12)) / sqrt(2.0)
        ),
        "theta": float(theta),
        "estimator": "user_hash_ab_with_single_snapshot_cuped",
    }


def _cuped_metric(control, treatment, preperiod, assigned, index):
    return _cuped_values(
        control[:, index], treatment[:, index], preperiod[:, index], assigned
    )


def _per_request(values, metric):
    requests = values[:, METRIC_INDEX["requests"]].clamp_min(1.0)
    return values[:, METRIC_INDEX[metric]] / requests


def _derived_experiment_metrics(pre, left, right, assigned):
    return {
        "negative_rate_per_request": _cuped_values(
            _per_request(left, "negative"),
            _per_request(right, "negative"),
            _per_request(pre, "negative"),
            assigned,
        ),
    }


def _experiment_metrics(config, preperiod, control, treatment, salt):
    pre = _concatenate(preperiod)
    left = _concatenate(control)
    right = _concatenate(treatment)
    user_id = torch.arange(config.users, device=left.device)
    assigned = assign_binary_torch(user_id, salt)
    if assigned.sum() == 0 or (~assigned).sum() == 0:
        raise ValueError("A/B assignment produced an empty cell")
    report = {
        name: _cuped_metric(left, right, pre, assigned, index)
        for index, name in enumerate(METRICS)
    }
    report.update(_derived_experiment_metrics(pre, left, right, assigned))
    return report


def _mixed_experiment_metrics(config, preperiod, mixed):
    pre = _concatenate(preperiod)
    observed = _concatenate(mixed.user_metrics)
    assigned = torch.cat(mixed.assigned_treatment)
    report = {
        name: _cuped_metric(
            observed, observed, pre, assigned, index
        )
        for index, name in enumerate(METRICS)
    }
    report.update(_derived_experiment_metrics(
        pre, observed, observed, assigned
    ))
    return report


def _paired_shadow_metrics(control, treatment):
    left = _concatenate(control)
    right = _concatenate(treatment)
    difference = right - left
    report = {}
    for index, name in enumerate(METRICS):
        effect = difference[:, index]
        standard_error = float(effect.std(unbiased=True) / sqrt(len(effect)))
        mean = float(effect.mean())
        report[name] = {
            "control_mean": float(left[:, index].mean()),
            "treatment_mean": float(right[:, index].mean()),
            "difference": mean,
            "standard_error": standard_error,
            "confidence_interval": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
            "estimator": "paired_full_rollout_shadow",
        }
    return report


def _arm_summary(run):
    return summarize_user_metrics(_concatenate(run.user_metrics))


def _release_shadow_world(run: TwinRun, device: torch.device) -> None:
    run.snapshot = None
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_twin_experiment(
    config: TwinConfig = TwinConfig(),
    control_policy: TwinPolicy = BASELINE_POLICY,
    treatment_policy: TwinPolicy | None = None,
    experiment_salt: int = 0x1B873593,
) -> TwinExperiment:
    treatment_policy = treatment_policy or TwinPolicy(
        name="history_aware_v2",
        realtime_weight=0.30,
        author_fatigue_penalty=0.06,
        cluster_fatigue_penalty=0.09,
        topic_fatigue_penalty=0.05,
    )
    kernel = DigitalTwinKernel(config)
    preperiod = kernel.preperiod(control_policy)
    return evaluate_from_preperiod(
        config, kernel, preperiod, control_policy, treatment_policy,
        experiment_salt,
    )


def evaluate_from_preperiod(
    config: TwinConfig,
    kernel: DigitalTwinKernel,
    preperiod: TwinRun,
    control_policy: TwinPolicy,
    treatment_policy: TwinPolicy,
    experiment_salt: int,
    trace_limit: int | None = None,
) -> TwinExperiment:
    if preperiod.snapshot is None:
        raise RuntimeError("experiment requires a materialized pre-period snapshot")
    control = kernel.arm(preperiod.snapshot, control_policy)
    control_summary = _arm_summary(control)
    _release_shadow_world(control, kernel.device)
    treatment = kernel.arm(preperiod.snapshot, treatment_policy)
    treatment_summary = _arm_summary(treatment)
    shadow_metrics = _paired_shadow_metrics(
        control.user_metrics, treatment.user_metrics
    )
    _release_shadow_world(treatment, kernel.device)
    mixed = run_mixed_world_ab(
        kernel, preperiod.snapshot, control_policy, treatment_policy,
        experiment_salt,
        trace_limit,
    )
    metrics = _mixed_experiment_metrics(
        config,
        preperiod.snapshot.preperiod_user_metrics,
        mixed,
    )
    trace_gates = {
        name: trace.validate() for name, trace in mixed.traces.items()
    }
    report = {
        "schema": "multi-surface-digital-twin-experiment-v1",
        "config": config.manifest(),
        "control": control_policy.manifest(),
        "treatment": treatment_policy.manifest(),
        "experiment_salt": experiment_salt,
        "preperiod": {
            "steps": config.preperiod_steps,
            "materializations": 1,
            "branching": "deep_snapshot_fork_after_preperiod",
            "recomputed_per_arm": False,
        },
        "control_summary": control_summary,
        "treatment_summary": treatment_summary,
        "cuped_ab": metrics,
        "paired_shadow_replay": shadow_metrics,
        "sample_evolution": mixed_sample_report(mixed),
        "ecosystem_interference": {
            "synthetic_lt_delta_vs_full_rollout": (
                metrics["synthetic_lt_measurement"]["difference"]
                - shadow_metrics["synthetic_lt_measurement"]["difference"]
            ),
            "stay_delta_vs_full_rollout": (
                metrics["stay_seconds"]["difference"]
                - shadow_metrics["stay_seconds"]["difference"]
            ),
        },
        "trace": {
            **{name: trace.manifest() for name, trace in mixed.traces.items()},
            "gates": trace_gates,
        },
        "supply": {
            "control_days": control.supply_days,
            "treatment_days": treatment.supply_days,
            "mixed_ab_days": mixed.supply_days,
        },
        "performance": {
            "preperiod_users_per_second": preperiod.users_per_second,
            "control_users_per_second": control.users_per_second,
            "treatment_users_per_second": treatment.users_per_second,
            "mixed_ab_users_per_second": mixed.users_per_second,
        },
        "evidence_boundary": (
            "Synthetic multi-agent engineering evidence calibrated by public "
            "data adapters; not production TikTok metrics or lift."
        ),
    }
    return TwinExperiment(report, preperiod, control, treatment, mixed)
