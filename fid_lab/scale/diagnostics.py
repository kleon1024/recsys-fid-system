"""Causal diagnostic ordering for offline AUC gains without online lift."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentDiagnostic:
    likely_causes: tuple[str, ...]
    first_checks: tuple[str, ...]


def diagnose_auc_without_lift(
    *,
    srm_p_value: float,
    trigger_rate: float,
    score_replay_delta: float,
    coarse_positive_pass: float,
    calibration_error: float,
    experiment_power: float,
) -> ExperimentDiagnostic:
    causes: list[str] = []
    checks: list[str] = []
    if srm_p_value < 0.01:
        causes.append("sample_ratio_mismatch")
        checks.append("assignment, eligibility, logging, and bot filtering")
    if trigger_rate < 0.01:
        causes.append("dilution_from_low_trigger_rate")
        checks.append("report intent-to-treat and triggered effect with fixed denominators")
    if score_replay_delta > 1e-5:
        causes.append("offline_online_prediction_skew")
        checks.append("FID, defaults, model manifest, runtime, and calibration replay")
    if coarse_positive_pass < 0.95:
        causes.append("cascade_opportunity_loss")
        checks.append("positive pass-through by recall route and coarse-rank slice")
    if calibration_error > 0.03:
        causes.append("value_fusion_miscalibration")
        checks.append("per-head calibration and value-tree marginal contribution")
    if experiment_power < 0.8:
        causes.append("underpowered_experiment")
        checks.append("clustered variance, MDE, duration, and positive-event count")
    if not causes:
        causes.append("metric_objective_mismatch_or_ecosystem_tradeoff")
        checks.append("slate diversity, novelty, guardrails, and long-term user or creator effects")
    return ExperimentDiagnostic(tuple(causes), tuple(checks))
