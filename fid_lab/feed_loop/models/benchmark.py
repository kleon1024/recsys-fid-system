"""Equal-log main-Feed model ladder with actual trajectory A/B."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ...evolution.evaluation.metrics import binary_metrics, grouped_auc
from ...simulation.ab import experiment_metrics, launch_decision, randomization_audit
from ...simulation.contracts import SimulationConfig
from ...simulation.environment import build_catalog
from ...simulation.experiment import build_feed_joiner
from ...simulation.policies import (
    GuardedBlendPolicy,
    HeuristicPolicy,
    fit_logistic_policy,
    fit_xgboost_policy,
    fit_xgboost_regression_policy,
)
from ...simulation.population import run_population
from .deep_policy import FeedDeepPolicy
from .artifact import publish_policy
from .multitask_policy import FeedMMoEPolicy


GUARDED_TOLERANCES = {
    "xgboost_stay_guarded_001": 0.01,
    "xgboost_stay_guarded_002": 0.02,
    "xgboost_stay_guarded_004": 0.04,
    "xgboost_stay_guarded_008": 0.08,
}


def _logging_examples(config, catalog):
    trajectories = run_population(
        config, catalog, HeuristicPolicy(), range(config.users), explore=True
    )
    rows = [row for trajectory in trajectories for row in trajectory.rows]
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    labels = np.asarray([row.response.long_view for row in rows], dtype=np.float32)
    task_labels = np.asarray(
        [
            (
                row.response.long_view,
                row.response.high_quality_long_view,
                row.response.negative_feedback,
                np.log1p(row.response.stay_seconds) / np.log(181.0),
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    users = np.asarray([row.user_id for row in rows], dtype=np.int64)
    sessions = np.asarray([row.session_id for row in rows], dtype=np.int8)
    propensities = np.asarray(
        [row.selection_probability for row in rows], dtype=np.float32
    )
    return rows, features, labels, task_labels, users, sessions, propensities


def _candidate_quality(rows, policy) -> dict[str, float]:
    candidate_features = np.asarray(
        [row.candidate_features for row in rows], dtype=np.float32
    )
    request_count, candidate_count, feature_count = candidate_features.shape
    scores = policy.score(candidate_features.reshape(-1, feature_count)).reshape(
        request_count, candidate_count
    )
    choice = scores.argmax(axis=1)
    probabilities = np.asarray(
        [row.candidate_oracle_long_view for row in rows], dtype=np.float32
    )
    selected = probabilities[np.arange(request_count), choice]
    oracle = probabilities.max(axis=1)
    return {
        "chosen_true_lt_probability": float(np.mean(selected)),
        "oracle_regret": float(np.mean(oracle - selected)),
    }


def _train_mmoe(train_x, train_y, validation_x, validation_y, epochs, device, seed):
    policy = FeedMMoEPolicy(train_x.shape[1], device, seed)
    started = perf_counter()
    policy.fit(
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs,
    )
    diagnostics = policy.diagnostics(validation_x)
    metadata = {
        "library": "pytorch",
        "train_seconds": perf_counter() - started,
        "parameters": policy.parameters,
        "loss_history": policy.loss_history,
        "gate_entropy": diagnostics.gate_entropy,
        "expert_utilization": diagnostics.expert_utilization,
        "propensity_correction": "clipped_ips_resampling",
    }
    return policy, metadata


def _train_deep_models(
    train_x,
    train_y,
    validation_x,
    validation_y,
    epochs,
    device,
    seed,
    names=("wide_deep", "deepfm", "dcnv2"),
):
    policies = []
    metadata = {}
    for name in names:
        policy = FeedDeepPolicy(name, device, seed)
        started = perf_counter()
        policy.fit(train_x, train_y, validation_x, validation_y, epochs)
        metadata[name] = {
            "library": "deepctr-torch-0.3.0",
            "train_seconds": perf_counter() - started,
            "parameters": policy.parameters,
            "loss_history": policy.loss_history,
            "propensity_correction": "clipped_ips_resampling",
        }
        policies.append(policy)
    return policies, metadata


def _joiner_evidence(report) -> dict[str, object]:
    artifact_ids = sorted(
        {
            example.manifest.get("artifact_id", "")
            for example in report.fine
            if example.manifest.get("artifact_id")
        }
    )
    return {
        "recall_examples": len(report.recall),
        "coarse_examples": len(report.coarse),
        "fine_examples": len(report.fine),
        "duplicate_events": report.duplicate_events,
        "orphan_events": report.orphan_events,
        "immature_task_labels": report.immature_task_labels,
        "artifact_ids": artifact_ids,
        "all_fine_examples_artifact_bound": bool(report.fine)
        and all(example.manifest.get("artifact_id") for example in report.fine),
    }


def _train_models(
    config, features, labels, task_labels, sessions, propensities,
    epochs,
    device,
    candidate_names,
):
    train_mask = sessions == 0
    validation_mask = sessions == 1
    test_mask = sessions >= 2
    train_x, train_y = features[train_mask], labels[train_mask]
    validation_x, validation_y = features[validation_mask], labels[validation_mask]
    weights = np.minimum(1.0 / np.maximum(propensities[train_mask], 1e-4), 20.0)
    lr = fit_logistic_policy(
        "lr_full_feed",
        train_x,
        train_y,
        tuple(range(features.shape[1])),
        config.seed,
        weights,
    )
    policies = [lr]
    training = {lr.name: {"library": "scikit-learn", "train_seconds": None}}
    for name, target in (
        ("xgboost", train_y),
        ("xgboost_quality", task_labels[train_mask, 1]),
    ):
        if name not in candidate_names:
            continue
        policy = fit_xgboost_policy(
            train_x, target, config.seed, weights, name=name
        )
        policies.append(policy)
        training[name] = {
            "library": "xgboost",
            "train_seconds": None,
            "target": "long_view" if name == "xgboost" else "high_quality_long_view",
            "propensity_correction": "clipped_ips_weight",
        }
    guarded_names = tuple(
        name for name in GUARDED_TOLERANCES if name in candidate_names
    )
    if "xgboost_stay" in candidate_names or guarded_names:
        stay_policy = fit_xgboost_regression_policy(
            train_x,
            task_labels[train_mask, 3],
            config.seed,
            weights,
        )
        if "xgboost_stay" in candidate_names:
            policies.append(stay_policy)
        training[stay_policy.name] = {
            "library": "xgboost",
            "train_seconds": None,
            "target": "normalized_log_stay_seconds",
            "propensity_correction": "clipped_ips_weight",
        }
        for guarded_name in guarded_names:
            tolerance = GUARDED_TOLERANCES[guarded_name]
            guarded = GuardedBlendPolicy(
                guarded_name,
                lr,
                stay_policy,
                config.candidates,
                base_score_tolerance=tolerance,
            )
            policies.append(guarded)
            training[guarded.name] = {
                "library": "constrained-serving-policy",
                "train_seconds": None,
                "target": "expected_stay_inside_lr_quality_feasible_set",
                "base_score_tolerance": tolerance,
            }
    if "xgboost_feed_value" in candidate_names:
        feed_value = (
            0.45 * task_labels[train_mask, 3]
            + 0.35 * task_labels[train_mask, 0]
            + 0.20 * task_labels[train_mask, 1]
            - 0.15 * task_labels[train_mask, 2]
        )
        value_policy = fit_xgboost_regression_policy(
            train_x,
            feed_value,
            config.seed,
            weights,
            name="xgboost_feed_value",
        )
        policies.append(value_policy)
        training[value_policy.name] = {
            "library": "xgboost",
            "train_seconds": None,
            "target": "0.45_log_stay+0.35_long+0.20_quality-0.15_negative",
            "propensity_correction": "clipped_ips_weight",
        }
    resample_rng = np.random.default_rng(config.seed + 91)
    resampled = resample_rng.choice(
        len(train_x), size=len(train_x), replace=True, p=weights / weights.sum()
    )
    deep_names = tuple(
        name for name in ("wide_deep", "deepfm", "dcnv2")
        if name in candidate_names
    )
    if deep_names:
        deep_policies, deep_training = _train_deep_models(
            train_x[resampled], train_y[resampled], validation_x, validation_y,
            epochs, device, config.seed, deep_names,
        )
        policies.extend(deep_policies)
        training.update(deep_training)
    if "mmoe_value_tree" in candidate_names:
        mmoe, mmoe_training = _train_mmoe(
            train_x[resampled], task_labels[train_mask][resampled, :3], validation_x,
            task_labels[validation_mask, :3], epochs, device, config.seed,
        )
        policies.append(mmoe)
        training[mmoe.name] = mmoe_training
    return policies, training, train_mask, validation_mask, test_mask, weights


def _offline_evidence(
    policies, training, rows, features, labels, user_ids, test_mask, signal_version,
    artifact_dir,
):
    test_features = features[test_mask]
    test_labels = labels[test_mask]
    test_users = user_ids[test_mask]
    test_rows = [row for row, selected in zip(rows, test_mask) if selected]
    published = {}
    replay_deltas = {}
    for policy in policies:
        serving, replay_delta = publish_policy(
            policy, test_features, signal_version, artifact_dir
        )
        published[policy.name] = serving
        replay_deltas[policy.name] = replay_delta
    offline = {}
    for policy in policies:
        scores = (
            policy.predict_tasks(test_features)["long_view"]
            if isinstance(policy, FeedMMoEPolicy)
            else policy.score(test_features)
        )
        offline[policy.name] = {
            **binary_metrics(test_labels, scores),
            "user_gauc": grouped_auc(test_labels, scores, test_users),
            "candidate": _candidate_quality(test_rows, policy),
            "shadow_replay_score_delta": replay_deltas[policy.name],
            "artifact_manifest": published[policy.name].artifact_manifest,
            **training[policy.name],
        }
    return offline, published, test_users


def _stateful_launches(config, items, ab_users, policies, published):
    evaluation_config = SimulationConfig(
        users=ab_users, items=items, joiner_users=min(ab_users, 100),
        seed=config.seed, signal_version=config.signal_version,
    )
    catalog = build_catalog(evaluation_config)
    experiment_users = np.arange(ab_users) + 50_000_000
    control_policy = published[policies[0].name]
    control = run_population(evaluation_config, catalog, control_policy, experiment_users)
    launches = {}
    for index, policy in enumerate(policies[1:]):
        treatment_policy = published[policy.name]
        treatment = run_population(
            evaluation_config, catalog, treatment_policy, experiment_users
        )
        assigned = np.random.default_rng(config.seed + 700 + index).random(ab_users) < 0.5
        metrics, potential = experiment_metrics(control, treatment, assigned)
        observed = [
            treatment[user] if assigned[user] else control[user]
            for user in range(ab_users)
        ]
        joined = build_feed_joiner(
            evaluation_config, catalog, observed,
            (control_policy, treatment_policy), assigned,
        )
        launches[f"lr_to_{policy.name}"] = {
            "metrics": metrics,
            "decision": launch_decision(metrics),
            "artifact_manifest": treatment_policy.artifact_manifest,
            "joiner": _joiner_evidence(joined),
            "randomization_audit": randomization_audit(
                potential, config.seed + 1700 + index
            ),
        }
    return launches


def run_feed_model_ladder(
    users: int = 1_000,
    items: int = 4_000,
    ab_users: int = 1_000,
    epochs: int = 12,
    device: str = "cpu",
    signal_version: str = "industrial-cross-sequence-v1",
    candidate_names: tuple[str, ...] = (
        "xgboost",
        "xgboost_quality",
        "xgboost_stay",
        "xgboost_feed_value",
        "xgboost_stay_guarded_002",
        "wide_deep",
        "deepfm",
        "dcnv2",
        "mmoe_value_tree",
    ),
    artifact_dir: str | None = None,
) -> dict[str, object]:
    config = SimulationConfig(
        users=users,
        items=items,
        joiner_users=0,
        signal_version=signal_version,
    )
    catalog = build_catalog(config)
    rows, features, labels, task_labels, user_ids, sessions, propensities = (
        _logging_examples(config, catalog)
    )
    policies, training, train_mask, validation_mask, test_mask, weights = _train_models(
        config, features, labels, task_labels, sessions, propensities, epochs, device,
        candidate_names,
    )
    offline, published, test_users = _offline_evidence(
        policies, training, rows, features, labels, user_ids, test_mask,
        signal_version, None if artifact_dir is None else Path(artifact_dir),
    )
    launches = _stateful_launches(config, items, ab_users, policies, published)
    return {
        "config": asdict(config),
        "signal_version": signal_version,
        "examples": len(rows),
        "positive_rate": float(labels.mean()),
        "split_contract": {
            "train": "session_0",
            "validation": "session_1",
            "test": "sessions_2_3_returning_users",
            "fresh_user_ab": "disjoint_cold_users",
            "train_examples": int(train_mask.sum()),
            "validation_examples": int(validation_mask.sum()),
            "test_examples": int(test_mask.sum()),
            "test_users": int(np.unique(test_users).size),
        },
        "propensity_effective_sample_size": float(
            weights.sum() ** 2 / np.square(weights).sum()
        ),
        "offline": offline,
        "launches": launches,
    }
