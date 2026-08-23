"""Request-level Local Search candidate and mature-label authority."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalSearchExamples:
    request_id: torch.Tensor
    poi_ids: torch.Tensor
    route_bits: torch.Tensor
    exposed_indices: torch.Tensor
    position_propensity: torch.Tensor
    labels: torch.Tensor
    label_masks: torch.Tensor
    served_scores: torch.Tensor

    def validate(self, exposed_candidates):
        requests, candidates = self.poi_ids.shape
        if self.exposed_indices.shape != (requests, exposed_candidates):
            raise ValueError("Local Search exposure shape mismatch")
        if self.labels.shape[:2] != (requests, candidates):
            raise ValueError("Local Search label shape mismatch")
        if self.labels.shape != self.label_masks.shape:
            raise ValueError("Local Search label mask shape mismatch")
        if not torch.isfinite(self.position_propensity).all():
            raise ValueError("Local Search propensity must be finite")
        if (self.position_propensity <= 0).any():
            raise ValueError("Local Search propensity must be positive")
