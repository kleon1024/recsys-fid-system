"""Zero-copy CUDA serving for a published LR plus XGBoost guarded policy."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import joblib
import torch
from xgboost import Booster

from ...models.artifact import feature_schema_hash
from ...tensor_cascade import stage_diagnostics
from ..calibration.nonlinear import nonlinear_stay_adjustment
from .features import build_tensor_features


V2_STAY_LOG_INTERCEPT_CALIBRATION = 0.25


def _selected_candidate(
    candidates, choice, user_ids, state, device, features, scores
):
    batch = torch.arange(len(user_ids), device=device)
    names = (
        "topics", "quality", "is_poi", "commerce", "poi_quality",
        "inventory", "same_city", "search_match", "retarget_match",
        "fulfillment", "candidate_topic", "item_ids", "content_type",
        "ad_value", "live_value", "duration",
    )
    selected = {name: candidates[name][batch, choice] for name in names}
    selected["fine_scores"] = scores
    selected["mix_scores"] = scores
    selected["fine_choice"] = choice
    selected["final_choice"] = choice
    if "stay_nonlinear" in candidates:
        selected["stay_nonlinear"] = candidates["stay_nonlinear"][batch, choice]
    else:
        selected["stay_nonlinear"] = (
            nonlinear_stay_adjustment(features[batch, choice])
            + V2_STAY_LOG_INTERCEPT_CALIBRATION
        )
    true_affinity = torch.einsum("bkd,bd->bk", candidates["topics"], state["interest"])
    true_utility = true_affinity + 0.45 * candidates["quality"]
    chosen_utility = true_utility[batch, choice]
    selected["coarse_oracle_survives"] = torch.ones_like(chosen_utility).bool()
    selected["coarse_pass_fraction"] = candidates["coarse_pass_fraction"]
    selected["oracle_regret"] = torch.maximum(
        candidates["merged_oracle_utility"], candidates["audit_oracle_utility"]
    ) - chosen_utility
    selected["poi_candidate_fraction"] = candidates["is_poi"].mean(dim=1)
    selected["organic_opportunity_cost"] = torch.zeros_like(chosen_utility)
    selected.update(
        stage_diagnostics(
            candidates,
            choice,
            choice,
            candidates["audit_oracle_in_coarse"],
        )
    )
    return selected


class TensorArtifactPolicy:
    eligible_fraction = 1.0
    observation_noise = 0.45
    local_observation_noise = 0.15
    realtime_interest_rate = 0.06
    multi_queue = False

    def __init__(
        self,
        report_path: Path,
        artifact_dir: Path,
        device: str,
        treatment: bool,
    ) -> None:
        report = json.loads(report_path.read_text())
        offline = report["offline"]
        candidate_name = next(name for name in offline if "guarded" in name)
        manifest = offline[candidate_name]["artifact_manifest"]
        self.name = candidate_name if treatment else "lr_full_feed_tensor_control"
        self.treatment = treatment
        self.manifest = manifest
        if manifest["feature_schema_sha256"] != feature_schema_hash():
            raise ValueError("tensor and published feature schemas do not match")
        if manifest["signal_version"] != "heterogeneous-nonlinear-v2":
            raise ValueError("tensor artifact runner requires nonlinear V2")
        self.device = torch.device(device)
        base_path = artifact_dir / "lr_full_feed.joblib"
        challenger_path = artifact_dir / "xgboost_stay.json"
        self._verify(base_path, manifest["base_artifact_id"])
        self._verify(challenger_path, manifest["challenger_artifact_id"])
        logistic = joblib.load(base_path)
        self.coefficients = torch.as_tensor(
            logistic.coef_[0], dtype=torch.float32, device=self.device
        )
        self.intercept = torch.tensor(
            float(logistic.intercept_[0]), dtype=torch.float32, device=self.device
        )
        self.booster = Booster(model_file=challenger_path)
        self.booster.set_param({"device": device})
        self.candidates = 20
        self.tolerance = float(manifest["base_score_tolerance"])

    @staticmethod
    def _verify(path: Path, artifact_id: str) -> None:
        actual = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
        if actual != artifact_id:
            raise ValueError(f"artifact hash mismatch: {path.name}")

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "treatment": self.treatment,
            "artifact_manifest": self.manifest,
        }

    def _xgboost_score(self, features: torch.Tensor) -> torch.Tensor:
        prediction = self.booster.inplace_predict(features)
        return torch.utils.dlpack.from_dlpack(prediction).to(self.device)

    def select_candidate(self, user_ids, state, candidates, device, step, config):
        features = build_tensor_features(config, user_ids, state, candidates, step)
        if features.shape[1] != self.candidates:
            raise ValueError("artifact candidate count does not match training")
        flat = features.reshape(-1, features.shape[-1]).contiguous()
        base = torch.sigmoid(flat @ self.coefficients + self.intercept).reshape(
            features.shape[:2]
        )
        if self.treatment:
            stay = self._xgboost_score(flat).reshape(features.shape[:2])
            eligible = base >= (base.max(dim=1, keepdim=True).values - self.tolerance)
            choice = stay.masked_fill(~eligible, -1e9).argmax(dim=1)
        else:
            choice = base.argmax(dim=1)
        return _selected_candidate(
            candidates, choice, user_ids, state, device, features,
            stay.masked_fill(~eligible, -1e9) if self.treatment else base,
        )


class TensorColumnLogisticPolicy:
    """Serve one published feature-group LR on canonical GPU features."""

    eligible_fraction = 1.0
    observation_noise = 0.45
    local_observation_noise = 0.15
    realtime_interest_rate = 0.06
    multi_queue = False

    def __init__(
        self,
        report_path: Path,
        artifact_dir: Path,
        group: str,
        device: str,
    ) -> None:
        report = json.loads(report_path.read_text())
        evidence = report["offline"][group]
        manifest = evidence["artifact_manifest"]
        if manifest["feature_schema_sha256"] != feature_schema_hash():
            raise ValueError("feature-group artifact schema does not match tensor features")
        path = artifact_dir / manifest["artifact_file"]
        TensorArtifactPolicy._verify(path, manifest["artifact_id"])
        model = joblib.load(path)
        self.device = torch.device(device)
        self.name = manifest["model_name"]
        self.group = group
        self.manifest = manifest
        self.columns = torch.as_tensor(
            manifest["feature_columns"], dtype=torch.long, device=self.device
        )
        self.coefficients = torch.as_tensor(
            model.coef_[0], dtype=torch.float32, device=self.device
        )
        self.intercept = torch.tensor(
            float(model.intercept_[0]), dtype=torch.float32, device=self.device
        )
        if len(self.columns) != len(self.coefficients):
            raise ValueError("feature-group columns and LR coefficients differ")

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "feature_group": self.group,
            "artifact_manifest": self.manifest,
        }

    def select_candidate(self, user_ids, state, candidates, device, step, config):
        features = build_tensor_features(config, user_ids, state, candidates, step)
        model_features = features.index_select(2, self.columns)
        scores = torch.sigmoid(
            model_features @ self.coefficients + self.intercept
        )
        choice = scores.argmax(dim=1)
        return _selected_candidate(
            candidates, choice, user_ids, state, device, features, scores
        )
