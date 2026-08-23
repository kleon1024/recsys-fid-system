"""Offline model ladder, online A/B ladder, and policy iteration."""

from __future__ import annotations

import numpy as np

from ...evolution.evaluation.metrics import binary_metrics, grouped_auc
from ..ab import experiment_metrics, launch_decision, randomization_audit
from ..environment import FEATURE_NAMES
from ..features import FEATURE_GROUP_COLUMNS
from ..policies import (
    HeuristicPolicy, PopularPolicy, fit_logistic_policy, fit_policies,
    serialized_replay_deltas,
)
from ..population import run_population
from .audits import candidate_policy_audit
from .samples import example_arrays, joiner_report


def fit_ladder(config, features, labels, user_ids, session_ids, propensities):
    split = int(len(labels) * 0.75)
    inverse_propensity = np.minimum(1.0 / np.maximum(propensities[:split], 1e-4), 20.0)
    policies = fit_policies(
        features[:split], labels[:split], config.seed, sample_weight=inverse_propensity
    )
    basic_lr = fit_logistic_policy(
        "lr_basic_features",
        features[:split],
        labels[:split],
        FEATURE_GROUP_COLUMNS["basic"],
        config.seed,
        inverse_propensity,
    )
    sequence_lr = fit_logistic_policy(
        "lr_plus_sequence",
        features[:split],
        labels[:split],
        FEATURE_GROUP_COLUMNS["sequence"],
        config.seed,
        inverse_propensity,
    )
    realtime_lr = fit_logistic_policy(
        "lr_plus_realtime",
        features[:split],
        labels[:split],
        FEATURE_GROUP_COLUMNS["realtime"],
        config.seed,
        inverse_propensity,
    )
    local_lr = fit_logistic_policy(
        "lr_plus_local_context",
        features[:split],
        labels[:split],
        FEATURE_GROUP_COLUMNS["local_context"],
        config.seed,
        inverse_propensity,
    )
    learned_policies = (basic_lr, sequence_lr, realtime_lr, local_lr, *policies)
    replay_deltas = serialized_replay_deltas(learned_policies, features[split:])
    ladder = (PopularPolicy(), HeuristicPolicy(), *learned_policies)
    offline = {}
    for policy in learned_policies:
        scores = policy.score(features[split:])
        offline[policy.name] = {
            **binary_metrics(labels[split:], scores),
            "user_gauc": grouped_auc(labels[split:], scores, user_ids[split:]),
            "session_gauc": grouped_auc(labels[split:], scores, session_ids[split:]),
            "feature_names": [
                FEATURE_NAMES[index]
                for index in (
                    policy.columns
                    if policy.columns is not None
                    else range(len(FEATURE_NAMES))
                )
            ],
        }
    return policies, ladder, offline, replay_deltas, inverse_propensity, split


def evaluate_ladder(config, catalog, ladder, final_policies):
    experiment_user_ids = np.arange(config.users) + 10_000_000
    trajectories = {
        policy.name: run_population(
            config, catalog, policy, experiment_user_ids
        )
        for policy in ladder
    }
    ab_ladder = {}
    assignments = {}
    final_assignment = None
    final_potential = None
    for launch_index, (control_policy, treatment_policy) in enumerate(
        zip(ladder, ladder[1:])
    ):
        assignment_rng = np.random.default_rng(config.seed + 77 + launch_index)
        assigned = assignment_rng.random(config.users) < 0.5
        launch_name = f"{control_policy.name}_to_{treatment_policy.name}"
        assignments[launch_name] = assigned
        control = trajectories[control_policy.name]
        treatment = trajectories[treatment_policy.name]
        metrics, potential = experiment_metrics(control, treatment, assigned)
        ab_ladder[launch_name] = {
            "control": control_policy.name,
            "treatment": treatment_policy.name,
            "assignment": {
                "control_users": int((~assigned).sum()),
                "treatment_users": int(assigned.sum()),
            },
            "metrics": metrics,
            "randomization_audit": randomization_audit(
                potential, config.seed + 991 + launch_index
            ),
            "decision": launch_decision(metrics),
        }
        if treatment_policy is final_policies[1]:
            final_assignment = assigned
            final_potential = potential
    if final_assignment is None or final_potential is None:
        raise RuntimeError("final LR-to-XGBoost launch was not evaluated")
    control = trajectories[final_policies[0].name]
    treatment = trajectories[final_policies[1].name]
    metrics = ab_ladder[
        f"{final_policies[0].name}_to_{final_policies[1].name}"
    ]["metrics"]
    final_randomization_audit = ab_ladder[
        f"{final_policies[0].name}_to_{final_policies[1].name}"
    ]["randomization_audit"]
    observed = [
        treatment[i] if final_assignment[i] else control[i]
        for i in range(config.users)
    ]
    return (
        ab_ladder,
        final_assignment,
        metrics,
        final_randomization_audit,
        observed,
        trajectories,
        assignments,
    )


def second_training_round(config, catalog, original_rows, ladder, trajectories, assignments):
    basic = next(policy for policy in ladder if policy.name == "lr_basic_features")
    sequence = next(policy for policy in ladder if policy.name == "lr_plus_sequence")
    launch_name = "lr_basic_features_to_lr_plus_sequence"
    assigned = assignments[launch_name]
    experiment_rows = [
        row
        for user_index, treatment_assigned in enumerate(assigned)
        for row in (
            trajectories[sequence.name][user_index].rows
            if treatment_assigned
            else trajectories[basic.name][user_index].rows
        )
    ]
    combined_rows = [*original_rows, *experiment_rows]
    features, labels, _, _, propensities = example_arrays(combined_rows, config)
    original_count = len(original_rows)
    weights = np.ones(len(combined_rows), dtype=np.float32)
    weights[:original_count] = np.minimum(
        1.0 / np.maximum(propensities[:original_count], 1e-4), 20.0
    )
    round_two = fit_logistic_policy(
        "lr_plus_sequence_round2",
        features,
        labels,
        (0, 1, 2, 3, 5, 8, 9, 4, 11),
        config.seed + 2,
        weights,
    )
    audit_users = np.arange(config.users) + 20_000_000
    audit_trajectories = run_population(
        config, catalog, HeuristicPolicy(), audit_users
    )
    audit_rows = [row for trajectory in audit_trajectories for row in trajectory.rows]
    audit_features = np.asarray(
        [row.features for row in audit_rows], dtype=np.float32
    )
    fresh_users = np.arange(config.users) + 30_000_000
    control = run_population(config, catalog, sequence, fresh_users)
    treatment = run_population(config, catalog, round_two, fresh_users)
    round_assignment = np.random.default_rng(config.seed + 404).random(config.users) < 0.5
    metrics, potential = experiment_metrics(control, treatment, round_assignment)
    observed = [
        treatment[index] if round_assignment[index] else control[index]
        for index in range(config.users)
    ]
    return {
        "training_examples_before": original_count,
        "mature_experiment_examples": len(experiment_rows),
        "training_examples_after": len(combined_rows),
        "candidate_policy_audit": candidate_policy_audit(
            audit_rows, (sequence, round_two)
        ),
        "shadow_replay_score_delta": serialized_replay_deltas(
            (round_two,), audit_features
        )[round_two.name],
        "ab_metrics": metrics,
        "randomization_audit": randomization_audit(potential, config.seed + 1404),
        "decision": launch_decision(metrics),
        "joiner": joiner_report(
            config, catalog, observed, (sequence, round_two), round_assignment
        ),
    }


