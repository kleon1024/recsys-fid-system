"""Closed-loop orchestration over samples, models, A/B, and review gates."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..ab import launch_decision
from ..contracts import SimulationConfig
from ..environment import build_catalog
from ..local.supply import run_supply_iteration
from .acceptance import simulator_acceptance
from .audits import behavior_distribution, candidate_policy_audit, cascade_audit
from .models import evaluate_ladder, fit_ladder, second_training_round
from .samples import joiner_report, training_data


def run_closed_loop_experiment(
    config: SimulationConfig = SimulationConfig(),
    include_local: bool = False,
) -> dict[str, object]:
    catalog = build_catalog(config)
    rows, features, labels, user_ids, session_ids, propensities = training_data(
        config, catalog
    )
    policies, ladder, offline, replay_deltas, inverse_propensity, split = fit_ladder(
        config, features, labels, user_ids, session_ids, propensities
    )
    (
        ab_ladder,
        final_assignment,
        metrics,
        randomization_audit,
        observed,
        trajectories,
        assignments,
    ) = evaluate_ladder(config, catalog, ladder, policies)
    policy_iteration = second_training_round(
        config, catalog, rows, ladder, trajectories, assignments
    )
    sequence_policy = next(
        policy for policy in ladder if policy.name == "lr_plus_sequence"
    )
    report = {
        "runtime": {
            "environment_contract": "Gymnasium",
            "reference_wheel": "sardine-rec==1.0.8",
            "protocol": ("request", "session", "cross_session"),
        },
        "config": asdict(config),
        "logging_policy": "quality/affinity rule with randomized exploration",
        "training_examples": len(rows),
        "long_view_prevalence": float(labels.mean()),
        "propensity": {
            "minimum": float(propensities.min()),
            "median": float(np.median(propensities)),
            "inverse_weight_effective_sample_size": float(
                inverse_propensity.sum() ** 2 / np.square(inverse_propensity).sum()
            ),
        },
        "offline": offline,
        "candidate_policy_audit": candidate_policy_audit(rows[split:], ladder),
        "cascade": cascade_audit(rows),
        "behavior_distribution": behavior_distribution(rows),
        "policy_runtime": {
            policy.name: {
                "training_device": policy.training_device,
                "serving_device": policy.serving_device,
            }
            for policy in policies
        },
        "shadow_replay_score_delta": replay_deltas,
        "offline_online_max_score_delta": max(replay_deltas.values()),
        "ab_assignment": {
            "control_users": int((~final_assignment).sum()),
            "treatment_users": int(final_assignment.sum()),
        },
        "ab_metrics": metrics,
        "ab_ladder": ab_ladder,
        "policy_iteration": policy_iteration,
        "single_experiment_truth_covered": all(
            metric["confidence_interval"][0]
            <= metric["true_itt"]
            <= metric["confidence_interval"][1]
            for metric in metrics.values()
        ),
        "randomization_audit": randomization_audit,
        "estimator_audit_passed": all(
            value["truth_inside_randomization_interval"]
            for value in randomization_audit.values()
        ),
        "launch_decision": launch_decision(metrics),
        "joiner": joiner_report(
            config, catalog, observed, policies, final_assignment
        ),
        "limitations": (
            "The environment validates mechanics and estimator recovery under explicit dynamics; "
            "it does not establish real production lift without logged-data calibration and a live randomized test."
        ),
    }
    if include_local:
        report["supply_iteration"] = run_supply_iteration(
            config,
            catalog,
            trajectories[sequence_policy.name],
            sequence_policy,
        )
    report["simulator_acceptance"] = simulator_acceptance(report)
    return report

