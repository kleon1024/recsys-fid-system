"""MMoE Feed policy for long-view, quality-view, and negative-feedback ranking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


TASKS = ("long_view", "high_quality_long_view", "negative_feedback")


class MMoENetwork(nn.Module):
    def __init__(self, inputs: int, experts: int = 4, expert_width: int = 48) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(inputs, 96),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(96, expert_width),
                nn.ReLU(),
            )
            for _ in range(experts)
        )
        self.gates = nn.ModuleDict(
            {task: nn.Linear(inputs, experts) for task in TASKS}
        )
        self.towers = nn.ModuleDict(
            {task: nn.Linear(expert_width, 1) for task in TASKS}
        )

    def forward(self, features: torch.Tensor):
        expert_values = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )
        logits = {}
        gates = {}
        for task in TASKS:
            gate = torch.softmax(self.gates[task](features), dim=1)
            mixture = (expert_values * gate.unsqueeze(-1)).sum(dim=1)
            logits[task] = self.towers[task](mixture).squeeze(1)
            gates[task] = gate
        return logits, gates


@dataclass(frozen=True)
class MMoEDiagnostics:
    gate_entropy: dict[str, float]
    expert_utilization: dict[str, list[float]]


class FeedMMoEPolicy:
    name = "mmoe_value_tree"

    def __init__(self, inputs: int, device: str, seed: int) -> None:
        torch.manual_seed(seed)
        self.inputs = inputs
        self.seed = seed
        self.device = torch.device(device)
        self.model = MMoENetwork(inputs).to(self.device)
        self.loss_history: list[float] = []

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
        epochs: int,
    ) -> None:
        features = torch.as_tensor(train_features, dtype=torch.float32)
        labels = torch.as_tensor(train_labels, dtype=torch.float32)
        validation_x = torch.as_tensor(validation_features, dtype=torch.float32).to(
            self.device
        )
        validation_y = torch.as_tensor(validation_labels, dtype=torch.float32).to(
            self.device
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        best_loss = float("inf")
        best_state = None
        rng = np.random.default_rng(20260823)
        for _ in range(epochs):
            order = rng.permutation(len(features))
            losses = []
            self.model.train()
            for start in range(0, len(order), 2_048):
                index = torch.as_tensor(order[start : start + 2_048])
                batch_x = features[index].to(self.device)
                batch_y = labels[index].to(self.device)
                logits, _ = self.model(batch_x)
                loss = self._loss(logits, batch_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            self.loss_history.append(float(np.mean(losses)))
            self.model.eval()
            with torch.no_grad():
                validation_logits, _ = self.model(validation_x)
                validation_loss = float(self._loss(validation_logits, validation_y))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)

    @staticmethod
    def _loss(logits, labels):
        task_weights = (1.0, 1.2, 0.25)
        losses = [
            weight
            * nn.functional.binary_cross_entropy_with_logits(logits[task], labels[:, index])
            for index, (task, weight) in enumerate(zip(TASKS, task_weights))
        ]
        return sum(losses)

    def predict_tasks(self, features: np.ndarray) -> dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            logits, _ = self.model(tensor)
        return {
            task: torch.sigmoid(value).cpu().numpy() for task, value in logits.items()
        }

    def score(self, features: np.ndarray) -> np.ndarray:
        tasks = self.predict_tasks(features)
        return (
            tasks["long_view"]
            + 0.8 * tasks["high_quality_long_view"]
            - 0.3 * tasks["negative_feedback"]
        )

    def diagnostics(self, features: np.ndarray) -> MMoEDiagnostics:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            _, gates = self.model(tensor)
        entropy = {}
        utilization = {}
        for task, gate in gates.items():
            entropy[task] = float((-(gate * gate.clamp_min(1e-8).log()).sum(dim=1)).mean())
            utilization[task] = [float(value) for value in gate.mean(dim=0).cpu()]
        return MMoEDiagnostics(entropy, utilization)

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def replay_delta(self, features: np.ndarray) -> float:
        before = self.score(features)
        replay = FeedMMoEPolicy(self.inputs, str(self.device), self.seed)
        replay.model.load_state_dict(deepcopy(self.model.state_dict()))
        return float(np.max(np.abs(before - replay.score(features))))
