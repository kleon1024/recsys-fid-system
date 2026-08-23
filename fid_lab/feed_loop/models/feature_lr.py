"""Train and publish isolated logistic-regression feature-group artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ...evolution.evaluation.metrics import binary_metrics, grouped_auc
from ...simulation.contracts import SimulationConfig
from ...simulation.environment import build_catalog
from ...simulation.features import feature_candidate_sets, feature_group_manifest
from ...simulation.policies import fit_logistic_policy
from .artifact import publish_policy
from .benchmark import candidate_quality, logging_examples


def train_feature_lr_suite(
    users: int,
    items: int,
    artifact_dir: Path,
    seed: int = 20260823,
) -> dict[str, object]:
    config = SimulationConfig(
        users=users,
        items=items,
        joiner_users=0,
        seed=seed,
        signal_version="heterogeneous-nonlinear-v2",
    )
    catalog = build_catalog(config)
    rows, features, labels, _, user_ids, sessions, propensities = logging_examples(
        config, catalog
    )
    train = sessions == 0
    validation = sessions == 1
    test = sessions >= 2
    weights = np.minimum(1.0 / np.maximum(propensities[train], 1e-4), 20.0)
    report = {}
    for offset, (group, columns) in enumerate(feature_candidate_sets().items()):
        policy = fit_logistic_policy(
            f"lr_feature_{group}",
            features[train],
            labels[train],
            columns,
            seed + offset,
            weights,
        )
        published, replay_delta = publish_policy(
            policy,
            features[validation],
            config.signal_version,
            artifact_dir,
        )
        scores = policy.score(features[test])
        test_rows = [row for row, selected in zip(rows, test) if selected]
        report[group] = {
            **binary_metrics(labels[test], scores),
            "user_gauc": grouped_auc(labels[test], scores, user_ids[test]),
            "candidate": candidate_quality(test_rows, policy),
            "artifact_manifest": dict(published.artifact_manifest),
            "shadow_replay_score_delta": replay_delta,
        }
    return {
        "suite": "feature-lr-v1",
        "config": asdict(config),
        "examples": len(rows),
        "split": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
        "feature_groups": feature_group_manifest(),
        "offline": report,
    }
