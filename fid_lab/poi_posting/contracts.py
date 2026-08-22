"""Single authority for the POI posting reconstruction's data and settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TASKS = ("select", "publish", "relevance")
PERMISSIONS = ("precise", "coarse", "ip_only")


@dataclass(frozen=True)
class PoiPostingConfig:
    seed: int = 43
    authors: int = 192
    pois: int = 160
    cities: int = 4
    categories: int = 8
    sessions: int = 1_600
    candidates_per_session: int = 8
    raw_semantic_dim: int = 12
    frames_per_draft: int = 4
    representation_dim: int = 32
    categorical_dim: int = 8
    experts: int = 3
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 0.003
    easy_negative_keep_rate: float = 0.25
    select_loss_weight: float = 0.5
    publish_loss_weight: float = 1.0
    relevance_loss_weight: float = 0.8


@dataclass(frozen=True)
class PostingBatch:
    session_id: np.ndarray
    event_time: np.ndarray
    author_id: np.ndarray
    poi_id: np.ndarray
    city_id: np.ndarray
    category_id: np.ndarray
    permission_id: np.ndarray
    frame_features: np.ndarray
    text_features: np.ndarray
    poi_features: np.ndarray
    numeric_features: np.ndarray
    labels: np.ndarray
    label_masks: np.ndarray
    hard_negative: np.ndarray

    def __len__(self) -> int:
        return int(self.author_id.shape[0])

    def take(self, indices: np.ndarray) -> PostingBatch:
        values = {
            name: getattr(self, name)[indices]
            for name in self.__dataclass_fields__
        }
        return PostingBatch(**values)
