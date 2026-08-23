"""Exact GPU serving adapters for every published V3 ladder artifact."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import joblib
import torch
from xgboost import Booster

from ...models.deep_policy import DENSE_INDICES, SPARSE_SPECS, FeedDeepPolicy
from ...models.feed_multitask import FeedMultiTaskPolicy
from ...models.multitask_policy import FeedMMoEPolicy
from ...tensor_cascade import _fine_score, coarse_rank, materialize_selected
from ...tensor_policies import PERSONALIZED
from ....value.predicted_tree import predicted_feed_value
from ..artifact.features import build_tensor_features


class TensorV3ModelPolicy:
    eligible_fraction = 1.0
    observation_noise = 0.12
    local_observation_noise = 0.12
    realtime_interest_rate = 0.06
    multi_queue = False

    def __init__(self, name, training_report, artifact_dir, device="cuda:0",
                 deployment_name=None, blend_weight=None, base_tolerance=None):
        report = json.loads(training_report.read_text())
        evidence = report["models"][name]
        manifest = evidence["artifact_manifest"]
        if manifest["signal_version"] != "kuairand-calibrated-v3":
            raise ValueError("V3 serving rejects a cross-epoch model")
        self.model_name = name
        self.name = deployment_name or name
        self.blend_weight = blend_weight
        self.base_tolerance = base_tolerance
        self.manifest = manifest
        self.device = torch.device(device)
        path = artifact_dir / manifest["artifact_file"]
        actual = f"sha256:{sha256(path.read_bytes()).hexdigest()}"
        if actual != manifest["artifact_id"]:
            raise ValueError(f"artifact hash mismatch: {name}")
        self.kind, self.model = self._load(path)

    def _load(self, path):
        if self.model_name == "lr_v3_long_view":
            model = joblib.load(path)
            parameters = (
                torch.as_tensor(
                    model.coef_[0], dtype=torch.float32, device=self.device
                ),
                torch.tensor(
                    float(model.intercept_[0]), dtype=torch.float32,
                    device=self.device,
                ),
            )
            return "lr", parameters
        if self.model_name == "xgboost_v3_long_view":
            model = Booster(model_file=path)
            model.set_param({"device": str(self.device)})
            return "xgboost", model
        if self.model_name in {"wide_deep", "deepfm", "dcnv2"}:
            wrapper = FeedDeepPolicy(
                self.model_name, str(self.device),
                task=self.manifest.get("prediction_task", "binary"),
            )
            wrapper.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            wrapper.model.eval()
            return "deepctr", wrapper
        if self.model_name == "mmoe_value_tree":
            wrapper = FeedMMoEPolicy(28, str(self.device), 20260823)
            wrapper.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            wrapper.model.eval()
            return "mmoe", wrapper
        if self.model_name.startswith("mmoe_feed_multitask"):
            wrapper = FeedMultiTaskPolicy(28, str(self.device), 20260823)
            wrapper.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            wrapper.model.eval()
            return "feed_multitask", wrapper
        raise ValueError(f"unsupported V3 model: {self.name}")

    def describe(self):
        return {
            "name": self.name,
            "artifact_manifest": self.manifest,
            "blend_weight": self.blend_weight,
            "base_tolerance": self.base_tolerance,
        }

    def _deepctr_input(self, features, wrapper):
        index = wrapper.model.feature_index
        width = max(end for _, end in index.values())
        values = torch.zeros(len(features), width, device=self.device)
        for field, column, vocabulary in SPARSE_SPECS:
            start, end = index[field]
            values[:, start:end] = torch.round(
                features[:, column : column + 1] * (vocabulary - 1)
            )
        for column in DENSE_INDICES:
            start, end = index[f"dense_{column}"]
            values[:, start:end] = features[:, column : column + 1]
        return values

    def _score_chunk(self, features):
        if self.kind == "lr":
            coefficients, intercept = self.model
            return torch.sigmoid(features @ coefficients + intercept)
        if self.kind == "xgboost":
            prediction = self.model.inplace_predict(features.contiguous())
            return torch.utils.dlpack.from_dlpack(prediction).to(self.device)
        if self.kind == "deepctr":
            return self.model.model(
                self._deepctr_input(features, self.model)
            ).flatten()
        if self.kind == "feed_multitask":
            return predicted_feed_value(
                self.model.predict_tasks_tensor(features),
                features,
                self.model.value_config,
            )
        logits, _ = self.model.model(features)
        tasks = {name: torch.sigmoid(value) for name, value in logits.items()}
        return (
            tasks["long_view"]
            + 0.8 * tasks["high_quality_long_view"]
            - 0.3 * tasks["negative_feedback"]
        )

    def _scores(self, features, chunk=250_000):
        output = []
        for start in range(0, len(features), chunk):
            output.append(self._score_chunk(features[start : start + chunk]))
        return torch.cat(output)

    def select_candidate(self, user_ids, state, candidates, device, step, config):
        features = build_tensor_features(config, user_ids, state, candidates, step)
        scores = self._scores(features.flatten(0, 1)).reshape(features.shape[:2])
        _, affinity = _fine_score(
            PERSONALIZED, state["eligible"], user_ids, state, candidates
        )
        coarse_scores, coarse_mask, coarse_keep = coarse_rank(
            PERSONALIZED, affinity, candidates, config.candidates
        )
        if self.blend_weight is not None and self.base_tolerance is not None:
            base_scores, _ = _fine_score(
                PERSONALIZED, state["eligible"], user_ids, state, candidates
            )
            normalized = (scores - scores.mean(dim=1, keepdim=True)) / (
                scores.std(dim=1, keepdim=True).clamp_min(1e-4)
            )
            coarse_base = base_scores.masked_fill(~coarse_mask, -torch.inf)
            eligible = coarse_mask & (
                coarse_base
                >= coarse_base.max(dim=1, keepdim=True).values - self.base_tolerance
            )
            scores = (base_scores + self.blend_weight * normalized).masked_fill(
                ~eligible, -1e9
            )
        else:
            scores = scores.masked_fill(~coarse_mask, -1e9)
        choice = scores.argmax(dim=1)
        return materialize_selected(
            self, user_ids, state, candidates, choice, choice, scores, scores,
            coarse_scores, coarse_mask, coarse_keep, device,
        )
