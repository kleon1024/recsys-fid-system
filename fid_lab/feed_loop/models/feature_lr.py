"""Train and publish isolated logistic-regression feature-group artifacts."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from ...evolution.evaluation.metrics import binary_metrics, grouped_auc
from ...simulation.contracts import SimulationConfig
from ...simulation.environment import build_catalog
from ...simulation.features import (
    campaign_candidate_sets,
    feature_campaign_manifest,
    feature_candidate_sets,
    feature_group_manifest,
)
from ...simulation.policies import fit_logistic_policy
from .artifact import publish_policy
from .benchmark import candidate_quality, logging_examples


def _train_candidates(config, artifact_dir, candidates):
    catalog = build_catalog(config)
    rows, features, labels, _, user_ids, sessions, propensities = logging_examples(
        config, catalog
    )
    train = sessions == 0
    validation = sessions == 1
    test = sessions >= 2
    weights = np.minimum(1.0 / np.maximum(propensities[train], 1e-4), 20.0)
    report = {}
    for offset, (group, columns) in enumerate(candidates.items()):
        policy = fit_logistic_policy(
            f"lr_feature_{group}",
            features[train],
            labels[train],
            columns,
            config.seed + offset,
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
    return rows, train, validation, test, report


def _config(users: int, items: int, seed: int) -> SimulationConfig:
    return SimulationConfig(
        users=users,
        items=items,
        joiner_users=0,
        seed=seed,
        signal_version="heterogeneous-nonlinear-v2",
    )


def _training_report(config, rows, train, validation, test, offline):
    return {
        "config": asdict(config),
        "examples": len(rows),
        "split": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
        "offline": offline,
    }


def _reuse_base_artifact(
    config,
    base_key,
    base_report_path,
    base_artifact_dir,
    artifact_dir,
):
    report = json.loads(base_report_path.read_text())
    if report["config"] != asdict(config):
        raise ValueError("campaign and active base training snapshots differ")
    evidence = report["offline"][base_key]
    manifest = evidence["artifact_manifest"]
    source = base_artifact_dir / manifest["artifact_file"]
    artifact_id = f"sha256:{sha256(source.read_bytes()).hexdigest()}"
    if artifact_id != manifest["artifact_id"]:
        raise ValueError("active base artifact hash mismatch")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, artifact_dir / source.name)
    return evidence


def train_feature_lr_suite(
    users: int,
    items: int,
    artifact_dir: Path,
    seed: int = 20260823,
) -> dict[str, object]:
    config = _config(users, items, seed)
    trained = _train_candidates(config, artifact_dir, feature_candidate_sets())
    return {
        "suite": "feature-lr-combination-suite-v2",
        "feature_groups": feature_group_manifest(),
        **_training_report(config, *trained),
    }


def train_feature_lr_campaign(
    campaign: str,
    users: int,
    items: int,
    artifact_dir: Path,
    seed: int = 20260823,
    base_report_path: Path | None = None,
    base_artifact_dir: Path | None = None,
) -> dict[str, object]:
    config = _config(users, items, seed)
    candidates = campaign_candidate_sets(campaign)
    campaign_manifest = feature_campaign_manifest(campaign)
    base_key = campaign_manifest["base_key"]
    base_columns = candidates.pop(base_key)
    if (base_report_path is None) != (base_artifact_dir is None):
        raise ValueError("base report and artifact directory must be provided together")
    if base_report_path:
        base_evidence = _reuse_base_artifact(
            config,
            base_key,
            base_report_path,
            base_artifact_dir,
            artifact_dir,
        )
    else:
        candidates = {base_key: base_columns, **candidates}
        base_evidence = None
    trained = _train_candidates(config, artifact_dir, candidates)
    rows, train, validation, test, offline = trained
    if base_evidence:
        offline = {base_key: base_evidence, **offline}
    return {
        "suite": "feature-lr-small-campaign-v1",
        "campaign": campaign_manifest,
        **_training_report(config, rows, train, validation, test, offline),
    }
