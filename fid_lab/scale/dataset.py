"""Concrete PyTorch batch contract and TensorFlow-equivalent schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import ScaleDataset


@dataclass(frozen=True)
class TensorSchema:
    name: str
    dtype: str
    shape: tuple[int | None, ...]


def tensor_schema() -> tuple[TensorSchema, ...]:
    """One authority used to document both PyTorch and tf.data tensors."""
    return (
        TensorSchema("sparse_fids", "int64", (None, 6)),
        TensorSchema("dense_features", "float32", (None, 10)),
        TensorSchema("history_item_ids", "int64", (None, 24)),
        TensorSchema("behavior_sequence", "float32", (None, 24, 8)),
        TensorSchema("sequence_mask", "bool", (None, 24)),
        TensorSchema("labels", "float32", (None, 6)),
        TensorSchema("label_masks", "bool", (None, 6)),
        TensorSchema("sample_weight", "float32", (None,)),
        TensorSchema("served_scores", "float32", (None, 4)),
    )


class FeedTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, dataset: ScaleDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset.examples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sparse_fids": torch.from_numpy(self.dataset.sparse_ids[index]),
            "dense_features": torch.from_numpy(self.dataset.dense_features[index]),
            "history_item_ids": torch.from_numpy(self.dataset.history_item_ids[index]),
            "behavior_sequence": torch.from_numpy(self.dataset.sequences[index]),
            "sequence_mask": torch.from_numpy(self.dataset.sequence_mask[index]),
            "labels": torch.from_numpy(self.dataset.labels[index]),
            "label_masks": torch.from_numpy(self.dataset.label_masks[index]),
            "sample_weight": torch.as_tensor(self.dataset.sample_weight[index]),
            "served_scores": torch.from_numpy(self.dataset.served_scores[index]),
        }


def tensorflow_generator(dataset: ScaleDataset):
    """Yield NumPy records accepted by tf.data.Dataset.from_generator."""
    for index in range(dataset.examples):
        yield {
            "sparse_fids": dataset.sparse_ids[index],
            "dense_features": dataset.dense_features[index],
            "history_item_ids": dataset.history_item_ids[index],
            "behavior_sequence": dataset.sequences[index],
            "sequence_mask": dataset.sequence_mask[index],
            "labels": dataset.labels[index],
            "label_masks": dataset.label_masks[index],
            "sample_weight": dataset.sample_weight[index],
            "served_scores": dataset.served_scores[index],
        }
