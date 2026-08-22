"""Equal-log main-Feed model ladder with actual trajectory A/B."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from ..evolution.evaluation.metrics import binary_metrics, grouped_auc
from ..simulation.ab import experiment_metrics, launch_decision, randomization_audit
from ..simulation.contracts import SimulationConfig
from ..simulation.environment import build_catalog
from ..simulation.policies import HeuristicPolicy, fit_logistic_policy
from ..simulation.population import run_population
from .deep_policy import FeedDeepPolicy
from .multitask_policy import FeedMMoEPolicy


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


def _train_deep_models(train_x, train_y, validation_x, validation_y, epochs, device, seed):
    policies = []
    metadata = {}
    for name in ("wide_deep", "deepfm", "dcnv2"):
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


def run_feed_model_ladder(
    users: int = 1_000,
    items: int = 4_000,
    ab_users: int = 1_000,
    epochs: int = 12,
    device: str = "cpu",
) -> dict[str, object]:
    config = SimulationConfig(users=users, items=items, joiner_users=0)
    catalog = build_catalog(config)
    rows, features, labels, task_labels, user_ids, sessions, propensities = (
        _logging_examples(config, catalog)
    )
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
    resample_rng = np.random.default_rng(config.seed + 91)
    resampled = resample_rng.choice(
        len(train_x),
        size=len(train_x),
        replace=True,
        p=weights / weights.sum(),
    )
    deep_policies, deep_training = _train_deep_models(
        train_x[resampled],
        train_y[resampled],
        validation_x,
        validation_y,
        epochs,
        device,
        config.seed,
    )
    policies.extend(deep_policies)
    training.update(deep_training)
    mmoe, mmoe_training = _train_mmoe(
        train_x[resampled],
        task_labels[train_mask][resampled],
        validation_x,
        task_labels[validation_mask],
        epochs,
        device,
        config.seed,
    )
    training[mmoe.name] = mmoe_training
    policies.append(mmoe)
    test_features = features[test_mask]
    test_labels = labels[test_mask]
    test_users = user_ids[test_mask]
    test_rows = [row for row, selected in zip(rows, test_mask) if selected]
    offline = {}
    for policy in policies:
        scores = (
            policy.predict_tasks(test_features)["long_view"]
            if isinstance(policy, FeedMMoEPolicy)
            else policy.score(test_features)
        )
        replay_delta = (
            policy.replay_delta(test_features)
            if isinstance(policy, (FeedDeepPolicy, FeedMMoEPolicy))
            else 0.0
        )
        offline[policy.name] = {
            **binary_metrics(test_labels, scores),
            "user_gauc": grouped_auc(test_labels, scores, test_users),
            "candidate": _candidate_quality(test_rows, policy),
            "shadow_replay_score_delta": replay_delta,
            **training[policy.name],
        }
    evaluation_config = SimulationConfig(
        users=ab_users, items=items, joiner_users=0, seed=config.seed
    )
    evaluation_catalog = build_catalog(evaluation_config)
    experiment_users = np.arange(ab_users) + 50_000_000
    control = run_population(evaluation_config, evaluation_catalog, lr, experiment_users)
    launches = {}
    for index, policy in enumerate(policies[1:]):
        treatment = run_population(
            evaluation_config, evaluation_catalog, policy, experiment_users
        )
        assigned = np.random.default_rng(config.seed + 700 + index).random(ab_users) < 0.5
        metrics, potential = experiment_metrics(control, treatment, assigned)
        launches[f"lr_to_{policy.name}"] = {
            "metrics": metrics,
            "decision": launch_decision(metrics),
            "randomization_audit": randomization_audit(
                potential, config.seed + 1700 + index
            ),
        }
    return {
        "config": asdict(config),
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
