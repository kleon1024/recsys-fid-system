"""Exact online adapter for V4 request-aware Feed rankers."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import torch

from .....value.predicted_tree import PredictedFeedValueConfig, predicted_feed_value
from ....tensor_cascade import _fine_score, coarse_rank, materialize_selected
from ....tensor_policies import PERSONALIZED
from ...artifact.features import build_tensor_features
from .contracts import TASKS
from .networks import MMoERanker, PLERanker, SingleTaskDIN, SingleTaskTransformer


REQUEST_MODELS = {
    "din": SingleTaskDIN,
    "transformer": SingleTaskTransformer,
    "mmoe": lambda: MMoERanker(len(TASKS)),
    "ple": lambda: PLERanker(len(TASKS)),
}


class TensorV4RequestPolicy:
    eligible_fraction = 1.0
    observation_noise = 0.12
    local_observation_noise = 0.12
    realtime_interest_rate = 0.06
    multi_queue = False

    def __init__(
        self,
        model_name: str,
        report_path: Path,
        artifact_dir: Path,
        device="cuda:0",
        blend_weight: float | None = None,
        base_tolerance: float = 0.05,
    ) -> None:
        if model_name not in REQUEST_MODELS:
            raise ValueError(f"unsupported V4 request model: {model_name}")
        report = json.loads(report_path.read_text())
        evidence = report["models"][model_name]
        artifact = evidence["artifact"]
        path = artifact_dir / artifact["path"]
        if sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("V4 request model artifact hash mismatch")
        self.device = torch.device(device)
        self.model_name = model_name
        self.name = (
            model_name if blend_weight is None
            else f"{model_name}_guarded_{blend_weight:g}"
        )
        self.blend_weight = blend_weight
        self.base_tolerance = base_tolerance
        self.behavior_world = report["behavior_world"]
        self.dataset_manifest_sha256 = report["dataset_manifest_sha256"]
        self.artifact = artifact
        self.value_config = PredictedFeedValueConfig()
        self.model = REQUEST_MODELS[model_name]().to(self.device)
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def describe(self):
        return {
            "name": self.name,
            "model_name": self.model_name,
            "artifact": self.artifact,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "behavior_world": self.behavior_world,
            "blend_weight": self.blend_weight,
            "base_tolerance": self.base_tolerance,
            "value_tree": self.value_config.manifest(),
        }

    def _predictions(self, logits):
        if self.model.task_count == 1:
            return {"long_view": torch.sigmoid(logits[:, :, 0])}
        tasks = {}
        for index, task in enumerate(TASKS):
            value = logits[:, :, index]
            value = (
                torch.sigmoid(value) if task.kind == "binary"
                else value.clamp(0.0, 1.0)
            )
            tasks["stay_norm" if task.name == "stay" else task.name] = value
        return tasks

    def _value(self, tasks, features, value_config=None):
        if self.model.task_count == 1:
            return tasks["long_view"]
        flat_features = features.flatten(0, 1)
        return predicted_feed_value(
            {name: value.flatten() for name, value in tasks.items()},
            flat_features, value_config or self.value_config,
        ).reshape(features.shape[:2])

    @torch.inference_mode()
    def predict_tasks(self, features, sequence, chunk=2_048):
        output = {}
        for start in range(0, len(features), chunk):
            stop = min(start + chunk, len(features))
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model(features[start:stop], sequence[start:stop])
                predictions = self._predictions(logits)
            for name, value in predictions.items():
                output.setdefault(name, []).append(value.float())
        return {name: torch.cat(parts) for name, parts in output.items()}

    def value(self, predictions, features):
        return self._value(predictions, features)

    def platform_value(self, predictions, features):
        """Feed-only value; business heads belong to the composite tree."""
        config = replace(
            self.value_config,
            local_anchor_weight=0.0,
            local_conversion_weight=0.0,
        )
        return self._value(predictions, features, config)

    @torch.inference_mode()
    def _scores(self, features, sequence, chunk=2_048):
        output = []
        for start in range(0, len(features), chunk):
            stop = min(start + chunk, len(features))
            with torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model(features[start:stop], sequence[start:stop])
                tasks = self._predictions(logits)
                output.append(self._value(tasks, features[start:stop]).float())
        return torch.cat(output)

    def select_candidate(self, user_ids, state, candidates, device, step, config):
        if "ranking_behavior_sequence" not in state:
            raise ValueError("V4 request serving requires the online ranking sequence")
        features = build_tensor_features(config, user_ids, state, candidates, step)
        scores = self._scores(features, state["ranking_behavior_sequence"])
        base, affinity = _fine_score(
            PERSONALIZED, state["eligible"], user_ids, state, candidates
        )
        coarse_scores, coarse_mask, coarse_keep = coarse_rank(
            PERSONALIZED, affinity, candidates, config.candidates
        )
        if self.blend_weight is None:
            served = scores.masked_fill(~coarse_mask, -1e9)
        else:
            eligible = coarse_mask & (
                base >= base.max(dim=1, keepdim=True).values - self.base_tolerance
            )
            normalized = (scores - scores.mean(dim=1, keepdim=True)) / (
                scores.std(dim=1, keepdim=True).clamp_min(1e-4)
            )
            served = (base + self.blend_weight * normalized).masked_fill(
                ~eligible, -1e9
            )
        choice = served.argmax(dim=1)
        return materialize_selected(
            self, user_ids, state, candidates, choice, choice, served, served,
            coarse_scores, coarse_mask, coarse_keep, device,
        )
