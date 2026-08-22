"""Small multi-task online trainer consuming joined examples and updating PS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import TrainingExample
from .parameter_server import ParameterSnapshot, UpdateResult, VersionedParameterServer


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class TrainResult:
    update: UpdateResult
    mean_loss: float
    examples: int


class OnlineMultiTaskTrainer:
    def __init__(self, server: VersionedParameterServer, learning_rate: float = 0.08) -> None:
        self.server = server
        self.learning_rate = learning_rate

    def vectorize(self, example: TrainingExample) -> np.ndarray:
        vector = np.zeros(self.server.feature_dim, dtype=np.float64)
        for field, bucket in enumerate(example.feature_buckets):
            index = (field * 131 + bucket) % self.server.feature_dim
            vector[index] += 1.0
        return vector

    def predict(
        self, examples: list[TrainingExample], snapshot: ParameterSnapshot | None = None
    ) -> np.ndarray:
        state = snapshot or self.server.snapshot()
        features = np.stack([self.vectorize(example) for example in examples])
        return sigmoid(features @ state.weights.T + state.bias)

    def train_microbatch(
        self, examples: list[TrainingExample], update_id: str
    ) -> TrainResult:
        if not examples:
            raise ValueError("training microbatch must not be empty")
        state = self.server.snapshot()
        features = np.stack([self.vectorize(example) for example in examples])
        labels = np.asarray(
            [[example.labels[task] for task in self.server.tasks] for example in examples]
        )
        weights = np.asarray([example.sample_weight for example in examples])[:, None]
        predictions = sigmoid(features @ state.weights.T + state.bias)
        error = (predictions - labels) * weights
        normalizer = max(float(weights.sum()), 1.0)
        weight_gradient = error.T @ features / normalizer
        bias_gradient = error.sum(axis=0) / normalizer
        epsilon = 1e-8
        loss = -weights * (
            labels * np.log(predictions + epsilon)
            + (1.0 - labels) * np.log(1.0 - predictions + epsilon)
        )
        update = self.server.apply(
            update_id,
            state.version,
            weight_gradient,
            bias_gradient,
            self.learning_rate,
        )
        return TrainResult(update, float(loss.mean()), len(examples))
