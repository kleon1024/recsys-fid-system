"""Shared-expert Feed model over primitive behavior, duration, and delayed heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ...value.predicted_tree import PredictedFeedValueConfig, predicted_feed_value


TASK_SPECS = {
    "play_3s": ("binary", 1, 0.35),
    "stay_norm": ("regression", 2, 1.00),
    "completion": ("regression", 3, 0.30),
    "long_view": ("binary", 5, 0.45),
    "quality_long_view": ("binary", 6, 0.55),
    "like": ("binary", 7, 0.20),
    "negative_feedback": ("binary", 8, 0.25),
    "returned_next_session": ("binary", 13, 0.15),
    "anchor_click": ("binary", 9, 0.10),
    "conversion": ("binary", 12, 0.05),
}


class FeedMultiTaskNetwork(nn.Module):
    def __init__(self, inputs: int, experts: int = 6, width: int = 64) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(inputs, 128), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(128, width), nn.ReLU(),
            )
            for _ in range(experts)
        )
        self.gates = nn.ModuleDict(
            {task: nn.Linear(inputs, experts) for task in TASK_SPECS}
        )
        self.towers = nn.ModuleDict(
            {task: nn.Linear(width, 1) for task in TASK_SPECS}
        )

    def forward(self, features):
        experts = torch.stack([expert(features) for expert in self.experts], dim=1)
        outputs = {}
        gates = {}
        for task in TASK_SPECS:
            gate = torch.softmax(self.gates[task](features), dim=1)
            outputs[task] = self.towers[task](
                (experts * gate.unsqueeze(-1)).sum(dim=1)
            ).squeeze(1)
            gates[task] = gate
        return outputs, gates


@dataclass(frozen=True)
class FeedMultiTaskDiagnostics:
    gate_entropy: dict[str, float]
    expert_utilization: dict[str, list[float]]


class FeedMultiTaskPolicy:
    name = "mmoe_feed_multitask_stay_v2"

    def __init__(self, inputs: int, device: str, seed: int = 20260823) -> None:
        torch.manual_seed(seed)
        self.inputs = inputs
        self.device = torch.device(device)
        self.seed = seed
        self.value_config = PredictedFeedValueConfig()
        self.model = FeedMultiTaskNetwork(inputs).to(self.device)
        self.loss_history: list[float] = []

    @staticmethod
    def targets(labels: np.ndarray) -> np.ndarray:
        values = []
        for task, (_, index, _) in TASK_SPECS.items():
            target = labels[:, index]
            if task == "stay_norm":
                target = target / 180.0
            values.append(target)
        return np.stack(values, axis=1).astype(np.float32)

    @staticmethod
    def _loss(outputs, targets):
        losses = []
        for index, (task, (kind, _, weight)) in enumerate(TASK_SPECS.items()):
            if kind == "binary":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs[task], targets[:, index]
                )
            else:
                loss = nn.functional.smooth_l1_loss(outputs[task], targets[:, index])
            losses.append(weight * loss)
        return sum(losses)

    def fit(self, train_features, train_labels, validation_features,
            validation_labels, epochs: int) -> None:
        train_x = torch.as_tensor(train_features, dtype=torch.float32)
        train_y = torch.as_tensor(self.targets(train_labels), dtype=torch.float32)
        validation_x = torch.as_tensor(
            validation_features, dtype=torch.float32, device=self.device
        )
        validation_y = torch.as_tensor(
            self.targets(validation_labels), dtype=torch.float32, device=self.device
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        best_loss = float("inf")
        best_state = None
        for _ in range(epochs):
            losses = []
            self.model.train()
            order = rng.permutation(len(train_x))
            for start in range(0, len(train_x), 4_096):
                index = torch.as_tensor(order[start:start + 4_096])
                features = train_x[index].to(self.device)
                targets = train_y[index].to(self.device)
                outputs, _ = self.model(features)
                loss = self._loss(outputs, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            self.loss_history.append(float(np.mean(losses)))
            self.model.eval()
            with torch.no_grad():
                outputs, _ = self.model(validation_x)
                validation_loss = float(self._loss(outputs, validation_y))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_tasks_tensor(self, features: torch.Tensor):
        outputs, _ = self.model(features)
        return {
            task: (torch.sigmoid(value) if kind == "binary" else value.clamp(0.0, 1.0))
            for task, value in outputs.items()
            for kind, _, _ in (TASK_SPECS[task],)
        }

    def predict_tasks(self, features: np.ndarray) -> dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            outputs = self.predict_tasks_tensor(tensor)
        return {task: value.cpu().numpy() for task, value in outputs.items()}

    def score(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            return predicted_feed_value(
                self.predict_tasks_tensor(tensor), tensor, self.value_config
            ).cpu().numpy()

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def replay_delta(self, features: np.ndarray) -> float:
        before = self.score(features)
        replay = FeedMultiTaskPolicy(self.inputs, str(self.device), self.seed)
        replay.model.load_state_dict(deepcopy(self.model.state_dict()))
        return float(np.max(np.abs(before - replay.score(features))))
