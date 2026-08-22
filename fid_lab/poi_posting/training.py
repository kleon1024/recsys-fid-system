"""Time-split training, negative sampling, and ranking evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as functional

from .contracts import PoiPostingConfig, PostingBatch, TASKS
from .model import PoiPostingRanker
from .synthetic import build_dataset


@dataclass(frozen=True)
class TaskMetric:
    auc: float
    average_precision: float
    positive_rate: float


@dataclass(frozen=True)
class ExperimentReport:
    examples: int
    train_examples_after_sampling: int
    hard_negatives: int
    task_metrics: dict[str, TaskMetric]
    baseline_ndcg_at_3: float
    model_ndcg_at_3: float
    model_recall_at_3: float
    mean_frame_attention_entropy: float
    config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["task_metrics"] = {
            task: asdict(metric) for task, metric in self.task_metrics.items()
        }
        return result


def tensor_batch(data: PostingBatch, indices: np.ndarray) -> dict[str, torch.Tensor]:
    integer_names = (
        "author_id",
        "poi_id",
        "city_id",
        "category_id",
        "permission_id",
    )
    float_names = (
        "frame_features",
        "text_features",
        "content_features",
        "poi_features",
        "numeric_features",
    )
    batch = {
        name: torch.as_tensor(getattr(data, name)[indices], dtype=torch.long)
        for name in integer_names
    }
    batch.update(
        {
            name: torch.as_tensor(getattr(data, name)[indices], dtype=torch.float32)
            for name in float_names
        }
    )
    return batch


def time_split(data: PostingBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sessions = np.unique(data.session_id)
    first = int(len(sessions) * 0.70)
    second = int(len(sessions) * 0.85)
    parts = (sessions[:first], sessions[first:second], sessions[second:])
    return tuple(np.flatnonzero(np.isin(data.session_id, part)) for part in parts)


def sampled_training_indices(
    data: PostingBatch, indices: np.ndarray, config: PoiPostingConfig
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed + 1)
    positive = data.labels[indices, 0] > 0
    hard = data.hard_negative[indices] > 0
    easy_keep = rng.random(len(indices)) < config.easy_negative_keep_rate
    keep = positive | hard | easy_keep
    selected = indices[keep]
    weights = np.ones(len(selected), dtype=np.float32)
    important = data.labels[selected, 0].astype(bool) | data.hard_negative[selected].astype(bool)
    weights[~important] = 1.0 / config.easy_negative_keep_rate
    return selected, weights


def task_positive_weights(data: PostingBatch, indices: np.ndarray) -> torch.Tensor:
    values: list[float] = []
    for task_index in range(len(TASKS)):
        mask = data.label_masks[indices, task_index] > 0
        labels = data.labels[indices, task_index][mask]
        positives = max(float(labels.sum()), 1.0)
        values.append(min(float((len(labels) - positives) / positives), 20.0))
    return torch.tensor(values, dtype=torch.float32)


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    sample_weights: torch.Tensor,
    positive_weights: torch.Tensor,
    task_weights: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for task_index, task in enumerate(TASKS):
        if task == "publish":
            probability = (
                torch.sigmoid(outputs["select"])
                * torch.sigmoid(outputs["publish"])
            ).clamp(1e-6, 1.0 - 1e-6)
            target = labels[:, task_index]
            loss = -(
                positive_weights[task_index] * target * torch.log(probability)
                + (1.0 - target) * torch.log1p(-probability)
            )
        else:
            loss = functional.binary_cross_entropy_with_logits(
                outputs[task],
                labels[:, task_index],
                reduction="none",
                pos_weight=positive_weights[task_index],
            )
        weighted_mask = masks[:, task_index] * sample_weights
        losses.append((loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0))
    return (torch.stack(losses) * task_weights).sum()


def train_model(
    data: PostingBatch,
    train_indices: np.ndarray,
    sample_weights: np.ndarray,
    config: PoiPostingConfig,
) -> PoiPostingRanker:
    torch.manual_seed(config.seed)
    model = PoiPostingRanker(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    positive_weights = task_positive_weights(data, train_indices)
    task_weights = torch.tensor(
        [
            config.select_loss_weight,
            config.publish_loss_weight,
            config.relevance_loss_weight,
        ]
    )
    rng = np.random.default_rng(config.seed + 2)
    for _ in range(config.epochs):
        order = rng.permutation(len(train_indices))
        for start in range(0, len(order), config.batch_size):
            positions = order[start : start + config.batch_size]
            indices = train_indices[positions]
            inputs = tensor_batch(data, indices)
            labels = torch.as_tensor(data.labels[indices], dtype=torch.float32)
            masks = torch.as_tensor(data.label_masks[indices], dtype=torch.float32)
            weights = torch.as_tensor(sample_weights[positions], dtype=torch.float32)
            outputs = model(inputs)
            total = multitask_loss(
                outputs,
                labels,
                masks,
                weights,
                positive_weights,
                task_weights,
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
    return model


def predict(
    model: PoiPostingRanker, data: PostingBatch, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch_indices = indices[start : start + 512]
            outputs = model(tensor_batch(data, batch_indices))
            select = torch.sigmoid(outputs["select"])
            publish = select * torch.sigmoid(outputs["publish"])
            relevance = torch.sigmoid(outputs["relevance"])
            task_scores = torch.stack([select, publish, relevance], dim=1)
            scores.append(task_scores.numpy())
            attention.append(data.frame_attention[batch_indices])
    return np.concatenate(scores), np.concatenate(attention)


def ranking_metrics(
    data: PostingBatch, indices: np.ndarray, score: np.ndarray
) -> tuple[float, float]:
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    sessions = data.session_id[indices]
    for session in np.unique(sessions):
        local = np.flatnonzero(sessions == session)
        labels = data.labels[indices[local], 0]
        if labels.sum() == 0:
            continue
        ranked = labels[np.argsort(-score[local])][:3]
        gains = ranked / np.log2(np.arange(2, 2 + len(ranked)))
        ndcg_values.append(float(gains.sum()))
        recall_values.append(float(ranked.sum() > 0))
    return float(np.mean(ndcg_values)), float(np.mean(recall_values))


def run_experiment(config: PoiPostingConfig = PoiPostingConfig()) -> ExperimentReport:
    data = build_dataset(config)
    train_full, _, test_indices = time_split(data)
    train_indices, sample_weights = sampled_training_indices(data, train_full, config)
    model = train_model(data, train_indices, sample_weights, config)
    scores, attention = predict(model, data, test_indices)
    metrics: dict[str, TaskMetric] = {}
    for task_index, task in enumerate(TASKS):
        mask = data.label_masks[test_indices, task_index] > 0
        labels = data.labels[test_indices, task_index][mask]
        task_scores = scores[:, task_index][mask]
        metrics[task] = TaskMetric(
            auc=float(roc_auc_score(labels, task_scores)),
            average_precision=float(average_precision_score(labels, task_scores)),
            positive_rate=float(labels.mean()),
        )
    model_score = scores[:, 1] * scores[:, 2]
    baseline_score = (
        -data.numeric_features[test_indices, 0]
        + 0.8 * data.numeric_features[test_indices, 1]
    )
    baseline_ndcg, _ = ranking_metrics(data, test_indices, baseline_score)
    model_ndcg, model_recall = ranking_metrics(data, test_indices, model_score)
    entropy = -(attention * np.log(np.maximum(attention, 1e-8))).sum(axis=1).mean()
    return ExperimentReport(
        examples=len(data),
        train_examples_after_sampling=len(train_indices),
        hard_negatives=int(data.hard_negative[train_full].sum()),
        task_metrics=metrics,
        baseline_ndcg_at_3=baseline_ndcg,
        model_ndcg_at_3=model_ndcg,
        model_recall_at_3=model_recall,
        mean_frame_attention_entropy=float(entropy),
        config=asdict(config),
    )
