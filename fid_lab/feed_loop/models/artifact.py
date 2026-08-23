"""Atomic serving copies and manifests for stateful Feed model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import joblib
import numpy as np
import torch
from xgboost import XGBClassifier, XGBRegressor

from ...simulation.environment import FEATURE_NAMES
from ...simulation.policies import (
    GuardedBlendPolicy,
    LearnedPolicy,
    LearnedRegressionPolicy,
)
from .deep_policy import DENSE_INDICES, SPARSE_SPECS, FeedDeepPolicy
from .multitask_policy import FeedMMoEPolicy


@dataclass(frozen=True)
class PublishedPolicy:
    policy: object
    artifact_manifest: Mapping[str, object]

    @property
    def name(self) -> str:
        return str(self.artifact_manifest["model_name"])

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.policy.score(features)


def feature_schema_hash() -> str:
    contract = {
        "feature_names": FEATURE_NAMES,
        "deep_dense_indices": DENSE_INDICES,
        "deep_sparse_specs": SPARSE_SPECS,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _device(policy, serving: bool = False) -> str:
    if isinstance(policy, LearnedPolicy):
        return policy.serving_device if serving else policy.training_device
    return str(policy.device)


def _save_and_reload(policy, directory: Path):
    if isinstance(policy, LearnedPolicy):
        if isinstance(policy.model, (XGBClassifier, XGBRegressor)):
            path = directory / f"{policy.name}.json"
            policy.model.save_model(path)
            model_type = XGBRegressor if isinstance(policy.model, XGBRegressor) else XGBClassifier
            loaded = model_type(device=policy.serving_device, n_jobs=1)
            loaded.load_model(path)
            artifact_format = "xgboost-json"
        else:
            path = directory / f"{policy.name}.joblib"
            joblib.dump(policy.model, path)
            loaded = joblib.load(path)
            artifact_format = "joblib"
        wrapper = (
            LearnedRegressionPolicy
            if isinstance(policy, LearnedRegressionPolicy)
            else LearnedPolicy
        )
        serving = wrapper(
            policy.name,
            loaded,
            policy.training_device,
            policy.serving_device,
            policy.columns,
        )
        return path, serving, artifact_format
    if isinstance(policy, FeedDeepPolicy):
        path = directory / f"{policy.name}.pt"
        torch.save(policy.model.state_dict(), path)
        serving = FeedDeepPolicy(policy.name, str(policy.device), policy.seed)
        serving.model.load_state_dict(
            torch.load(path, map_location=policy.device, weights_only=True)
        )
        return path, serving, "torch-state-dict"
    if isinstance(policy, FeedMMoEPolicy):
        path = directory / f"{policy.name}.pt"
        torch.save(policy.model.state_dict(), path)
        serving = FeedMMoEPolicy(policy.inputs, str(policy.device), policy.seed)
        serving.model.load_state_dict(
            torch.load(path, map_location=policy.device, weights_only=True)
        )
        return path, serving, "torch-state-dict"
    raise TypeError(f"unsupported policy artifact type: {type(policy).__name__}")


def publish_policy(
    policy,
    audit_features: np.ndarray,
    signal_version: str,
    artifact_dir: Path | None = None,
) -> tuple[PublishedPolicy, float]:
    if isinstance(policy, GuardedBlendPolicy):
        base, _ = publish_policy(
            policy.base, audit_features, signal_version, artifact_dir
        )
        challenger, _ = publish_policy(
            policy.challenger, audit_features, signal_version, artifact_dir
        )
        serving = GuardedBlendPolicy(
            policy.name,
            base.policy,
            challenger.policy,
            policy.candidates,
            policy.base_score_tolerance,
        )
        payload = {
            "model_name": policy.name,
            "base_artifact_id": base.artifact_manifest["artifact_id"],
            "challenger_artifact_id": challenger.artifact_manifest["artifact_id"],
            "base_score_tolerance": str(policy.base_score_tolerance),
            "feature_schema_sha256": feature_schema_hash(),
            "signal_version": signal_version,
        }
        artifact_hash = sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        manifest = {
            "artifact_id": f"sha256:{artifact_hash}",
            "artifact_format": "composite-manifest",
            "training_device": "cuda+cpu",
            "serving_device": "cpu",
            **payload,
        }
        delta = float(
            np.max(np.abs(policy.score(audit_features) - serving.score(audit_features)))
        )
        return PublishedPolicy(serving, manifest), delta
    before = policy.score(audit_features)
    if artifact_dir is None:
        with TemporaryDirectory() as temporary:
            path, serving, artifact_format = _save_and_reload(
                policy, Path(temporary)
            )
            artifact_hash = sha256(path.read_bytes()).hexdigest()
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path, serving, artifact_format = _save_and_reload(policy, artifact_dir)
        artifact_hash = sha256(path.read_bytes()).hexdigest()
    after = serving.score(audit_features)
    replay_delta = float(np.max(np.abs(before - after)))
    manifest = {
        "artifact_id": f"sha256:{artifact_hash}",
        "model_name": policy.name,
        "feature_schema_sha256": feature_schema_hash(),
        "signal_version": signal_version,
        "artifact_format": artifact_format,
        "artifact_file": path.name,
        "training_device": _device(policy),
        "serving_device": _device(serving, serving=True),
    }
    if isinstance(policy, LearnedPolicy):
        columns = policy.columns or tuple(range(len(FEATURE_NAMES)))
        manifest["feature_columns"] = tuple(columns)
        manifest["feature_names"] = tuple(FEATURE_NAMES[index] for index in columns)
    return PublishedPolicy(serving, manifest), replay_delta
