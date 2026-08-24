"""Launch Review for the unified Feed and Local serving graph."""

from __future__ import annotations

from dataclasses import asdict

from ...launches.experiment_protocol import (
    ExperimentPlan,
    phase_decision,
)
from ..governance.review import evaluate_governance_launch
from ..scale.experiment.trigger import (
    combine_tensor_ab,
    combine_tensor_counterfactual_ab,
    combine_tensor_cuped_ab,
)
from ..scale.tensor_engine import run_tensor_feed
from .composite import CompositeTensorPolicy
from .contracts import CompositeLaunchThresholds


def _shadow_gates(metrics, thresholds):
    return {
        "local_primary_positive": metrics["anchor_click_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_direction_nonnegative": (
            metrics["lt_value_per_user"]["treatment_mean"]
            >= metrics["lt_value_per_user"]["control_mean"]
        ),
        "platform_lt_noninferior": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= thresholds.shadow_lt_noninferiority,
        "stay_guardrail": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= thresholds.shadow_stay_noninferiority,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= thresholds.shadow_negative_upper,
        "coarse_recall_noninferior": metrics["coarse_feed_oracle_recall"][
            "confidence_interval"
        ][0] >= thresholds.shadow_coarse_recall_noninferiority,
    }


def _online_gates(metrics, thresholds):
    """Gate only on metrics observable in a production randomized A/B.

    Simulator oracle recall remains a shadow diagnostic. Treating it as an
    online gate would make the synthetic world leak into the experiment
    authority and could not be reproduced in production.
    """
    return {
        "local_primary_positive": metrics["anchor_click_rate"][
            "confidence_interval"
        ][0] > 0,
        "platform_lt_direction_nonnegative": (
            metrics["lt_value_per_user"]["treatment_mean"]
            >= metrics["lt_value_per_user"]["control_mean"]
        ),
        "platform_lt_noninferior": metrics["lt_value_per_user"][
            "confidence_interval"
        ][0] >= thresholds.online_lt_noninferiority,
        "stay_noninferior": metrics["stay_per_exposure"][
            "confidence_interval"
        ][0] >= thresholds.online_stay_noninferiority,
        "negative_guardrail": metrics["negative_rate"][
            "confidence_interval"
        ][1] <= thresholds.online_negative_upper,
    }


def _decision(shadow_gates, online_gates):
    if all(shadow_gates.values()) and all(online_gates.values()):
        return "pass"
    if all(shadow_gates.values()) and online_gates["local_primary_positive"]:
        return "continue_powered_online_experiment"
    return "hold_or_reject"


def _scenario_manifest(config, warmup_steps, behavior_world):
    runtime = asdict(config)
    for name in ("users", "batch_users", "device", "experiment_salt"):
        runtime.pop(name)
    return {
        "config": runtime,
        "measurement_start_step": warmup_steps,
        "behavior_world": behavior_world.describe(),
    }


def _control_policy(
    feed_policy, control_local_bundle, value_config, governance_config=None,
):
    return (
        feed_policy if control_local_bundle is None
        else CompositeTensorPolicy(
            feed_policy, control_local_bundle, value_config,
            governance_config=governance_config,
        )
    )


def run_composite_serving_launch(
    config, feed_policy, local_bundle, behavior_world, value_config=None,
    control_local_bundle=None, treatment_coarse_local_bundle=None,
    warmup_steps=0, thresholds=None, treatment_governance_config=None,
    control_governance_config=None, governance_thresholds=None,
    experiment_plan: ExperimentPlan | None = None,
):
    if experiment_plan is None:
        raise ValueError("a pre-registered experiment plan is required")
    thresholds = thresholds or CompositeLaunchThresholds()
    treatment = CompositeTensorPolicy(
        feed_policy, local_bundle, value_config, treatment_coarse_local_bundle,
        governance_config=treatment_governance_config,
    )
    control = _control_policy(
        feed_policy, control_local_bundle, value_config,
        control_governance_config,
    )
    scenario = _scenario_manifest(config, warmup_steps, behavior_world)
    experiment_plan.validate_run(
        control.describe(), treatment.describe(), scenario,
        config.users, config.experiment_salt,
    )
    if not 0 <= warmup_steps < config.steps:
        raise ValueError("composite warmup must precede the measurement window")
    control_schedule = tuple(control for _ in range(config.steps))
    treatment_schedule = (
        tuple(control for _ in range(warmup_steps))
        + tuple(treatment for _ in range(config.steps - warmup_steps))
    )
    control_world = run_tensor_feed(
        config, control, policy_schedule=control_schedule,
        measurement_start_step=warmup_steps, behavior_world=behavior_world,
    )
    treatment_world = run_tensor_feed(
        config, treatment, policy_schedule=treatment_schedule,
        measurement_start_step=warmup_steps, behavior_world=behavior_world,
    )
    paired = combine_tensor_counterfactual_ab(
        control_world, treatment_world
    )
    online = combine_tensor_ab(control_world, treatment_world)
    cuped = combine_tensor_cuped_ab(control_world, treatment_world)
    governance_experiment = (
        treatment_governance_config != control_governance_config
    )
    if governance_experiment:
        review = evaluate_governance_launch(
            paired, cuped, treatment_governance_config,
            config.steps - warmup_steps, config.users, governance_thresholds,
        )
        schema = "content-governance-launch-v1"
        online_diagnostics = {
            "predicted_integrity_risk_per_exposure": cuped[
                "predicted_integrity_risk_per_exposure"
            ],
            "near_duplicate_rate": cuped["near_duplicate_rate"],
            "governance_eligible_fraction": cuped[
                "governance_eligible_fraction"
            ],
        }
    else:
        shadow_gates = _shadow_gates(paired, thresholds)
        online_gates = _online_gates(cuped, thresholds)
        review = {
            "launch_thresholds": thresholds.manifest(),
            "shadow_gates": shadow_gates,
            "online_gates": online_gates,
            "decision": _decision(shadow_gates, online_gates),
        }
        schema = "unified-feed-business-serving-launch-v3"
        online_diagnostics = {
            "coarse_feed_oracle_recall": cuped["coarse_feed_oracle_recall"],
            "fine_oracle_regret_per_exposure": cuped[
                "fine_oracle_regret_per_exposure"
            ],
        }
    statistical_decision = review["decision"]
    review["statistical_decision"] = statistical_decision
    review["decision"] = phase_decision(
        experiment_plan, statistical_decision, (config.experiment_salt,)
    )
    return {
        "schema": schema,
        "config": asdict(config),
        "control": control.describe(),
        "treatment": treatment.describe(),
        "paired_shadow_replay": paired,
        "online_disjoint_ab": online,
        "online_cuped_ab": cuped,
        "warmup_steps": warmup_steps,
        "experiment_plan": experiment_plan.manifest(),
        "experiment_plan_fingerprint": experiment_plan.plan_fingerprint,
        **review,
        "online_diagnostics": online_diagnostics,
        "candidate_graph": treatment_world["candidate_graph"],
        "performance": {
            "control": control_world["performance"],
            "treatment": treatment_world["performance"],
        },
        "behavior_world": treatment_world["behavior_world"],
        "evidence_boundary": (
            "Unified main-Feed and Local serving in the external V5 simulator. "
            "Governance gates use served predictions and mature behavior only; "
            "hidden DGP quality remains diagnostic-only. This is synthetic "
            "Launch Review evidence, not production impact."
        ),
    }
