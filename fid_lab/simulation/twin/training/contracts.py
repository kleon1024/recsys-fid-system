"""Tensor contracts for observed events and the three sample authorities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..exchange import TASKS
from ..serving.surfaces import CANDIDATE_FEATURES
from ..platform.fids import TWIN_FID_SCHEMA_VERSION


SAMPLE_SCHEMA_VERSION = "twin-request-samples-v2-sparse-sequence"
FEATURE_SCHEMA_VERSION = "twin-observed-candidate-features-v2"
JOINER_VERSION = "twin-point-in-time-joiner-v2"

# One authority for simulated label availability. Units are twin steps.
TASK_MATURITY_STEPS = {
    "play": 0, "play_3s": 0, "long_view": 1, "complete": 1,
    "like": 1, "comment": 2, "share": 2, "follow": 4,
    "click": 0, "detail": 1, "favorite": 2, "add_cart": 2,
    "order": 8, "payment": 16, "negative": 1, "create": 2,
    "publish": 8,
}


@dataclass(frozen=True)
class SampleManifest:
    world_version: str
    served_policy: str
    experiment_cell: str
    watermark_step: int
    schema_version: str = SAMPLE_SCHEMA_VERSION
    feature_version: str = FEATURE_SCHEMA_VERSION
    joiner_version: str = JOINER_VERSION
    fid_schema_version: str = TWIN_FID_SCHEMA_VERSION


@dataclass(frozen=True)
class TwinEventBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    step: torch.Tensor
    surface: torch.Tensor
    request_sampling_probability: torch.Tensor
    served_policy_id: torch.Tensor
    experiment_cell_id: torch.Tensor
    candidate_item_ids: torch.Tensor
    candidate_kind: torch.Tensor
    route: torch.Tensor
    candidate_features: torch.Tensor
    candidate_sparse_fids: torch.Tensor
    candidate_sparse_buckets: torch.Tensor
    recall_score: torch.Tensor
    coarse_score: torch.Tensor
    fine_score: torch.Tensor
    eligible: torch.Tensor
    exposed_item_ids: torch.Tensor
    exposed_propensity: torch.Tensor
    selected_item_id: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    history_item_ids: torch.Tensor
    history_kinds: torch.Tensor
    history_surfaces: torch.Tensor
    history_steps: torch.Tensor
    manifest: SampleManifest

    @property
    def requests(self) -> int:
        return len(self.request_id)


@dataclass(frozen=True)
class RecallExampleBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    served_policy_id: torch.Tensor
    experiment_cell_id: torch.Tensor
    request_sampling_probability: torch.Tensor
    positive_item_id: torch.Tensor
    candidate_item_ids: torch.Tensor
    route: torch.Tensor
    negative_mask: torch.Tensor
    sampling_probability: torch.Tensor
    behavior_strength: torch.Tensor
    manifest: SampleManifest


@dataclass(frozen=True)
class CoarseRankExampleBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    served_policy_id: torch.Tensor
    experiment_cell_id: torch.Tensor
    request_sampling_probability: torch.Tensor
    item_ids: torch.Tensor
    route: torch.Tensor
    features: torch.Tensor
    recall_score: torch.Tensor
    served_coarse_score: torch.Tensor
    teacher_fine_score: torch.Tensor
    eligible: torch.Tensor
    exposed: torch.Tensor
    relevance: torch.Tensor
    sampling_probability: torch.Tensor
    manifest: SampleManifest


@dataclass(frozen=True)
class FineRankExampleBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    step: torch.Tensor
    surface: torch.Tensor
    served_policy_id: torch.Tensor
    experiment_cell_id: torch.Tensor
    request_sampling_probability: torch.Tensor
    item_ids: torch.Tensor
    item_kinds: torch.Tensor
    route: torch.Tensor
    positions: torch.Tensor
    sparse_fids: torch.Tensor
    sparse_buckets: torch.Tensor
    examination_propensity: torch.Tensor
    features: torch.Tensor
    served_score: torch.Tensor
    labels: torch.Tensor
    label_mask: torch.Tensor
    selected: torch.Tensor
    history_item_ids: torch.Tensor
    history_kinds: torch.Tensor
    history_surfaces: torch.Tensor
    history_steps: torch.Tensor
    manifest: SampleManifest


@dataclass(frozen=True)
class TrainingAuthorities:
    recall: RecallExampleBatch
    coarse: CoarseRankExampleBatch
    fine: FineRankExampleBatch

    def manifest(self) -> dict[str, object]:
        return {
            "schema": SAMPLE_SCHEMA_VERSION,
            "features": list(CANDIDATE_FEATURES),
            "tasks": list(TASKS),
            "fid_schema": self.fine.manifest.fid_schema_version,
            "sparse_fid_fields": self.fine.sparse_fids.shape[-1],
            "sequence_length": self.fine.history_item_ids.shape[-1],
            "recall_requests": len(self.recall.request_id),
            "coarse_requests": len(self.coarse.request_id),
            "fine_requests": len(self.fine.request_id),
            "watermark_step": self.fine.manifest.watermark_step,
            "logging_policy_versions": int(
                torch.unique(self.fine.served_policy_id).numel()
            ),
            "experiment_cells": int(
                torch.unique(self.fine.experiment_cell_id).numel()
            ),
        }
