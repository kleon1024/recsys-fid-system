"""Request-level tensors for learned DGP training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class WorldModelSplit:
    selected_features: torch.Tensor
    slate_features: torch.Tensor
    sequence: torch.Tensor
    labels: torch.Tensor
    label_masks: torch.Tensor
    weights: torch.Tensor
    lifecycle: torch.Tensor
    region: torch.Tensor
    user_ids: torch.Tensor
    request_steps: torch.Tensor
    exposed_index: torch.Tensor
    candidate_fine_scores: torch.Tensor
    candidate_audit_utility: torch.Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "selected_features": self.selected_features[indices].to(device, non_blocking=True),
            "slate_features": self.slate_features[indices].to(device, non_blocking=True),
            "sequence": self.sequence[indices].to(device, non_blocking=True),
            "labels": self.labels[indices].to(device, non_blocking=True),
            "label_masks": self.label_masks[indices].to(device, non_blocking=True),
            "weights": self.weights[indices].to(device, non_blocking=True),
            "lifecycle": self.lifecycle[indices].to(device, non_blocking=True),
            "region": self.region[indices].to(device, non_blocking=True),
        }


def load_world_split(dataset_dir: Path, split: str, max_rows: int | None = None) -> WorldModelSplit:
    payload = torch.load(
        dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False
    )["tensors"]
    rows = len(payload["exposed_index"])
    if max_rows is not None:
        rows = min(rows, max_rows)
    row_index = torch.arange(rows)
    choice = payload["exposed_index"][:rows].long()
    propensity = payload["exposure_propensity"][:rows].float().clamp_min(1e-4)
    weights = (1.0 / propensity).clamp_max(20.0)
    weights /= weights.mean()
    labels, label_masks = _with_session_exit(payload, rows)
    return WorldModelSplit(
        selected_features=payload["candidate_features"][:rows][row_index, choice].float(),
        slate_features=payload["candidate_features"][:rows].float(),
        sequence=payload["behavior_sequence"][:rows].float(),
        labels=labels,
        label_masks=label_masks,
        weights=weights,
        lifecycle=payload["lifecycle_bucket"][:rows].long(),
        region=payload["region_bucket"][:rows].long(),
        user_ids=payload["user_id"][:rows].long(),
        request_steps=payload["request_step"][:rows].long(),
        exposed_index=choice,
        candidate_fine_scores=payload["candidate_fine_scores"][:rows].float(),
        candidate_audit_utility=payload["candidate_audit_utility"][:rows].float(),
    )


def _with_session_exit(payload, rows):
    labels = payload["labels"][:rows].float()
    masks = payload["label_masks"][:rows].float()
    users = payload["user_id"][:rows].long()
    steps = payload["request_step"][:rows].long()
    sessions = payload["session_id"][:rows].long()
    order = torch.argsort(users * 1_000_000 + steps, stable=True)
    ordered_users = users[order]
    next_observed = torch.zeros(rows, dtype=torch.bool)
    exits = torch.zeros(rows)
    same_user = ordered_users[:-1] == ordered_users[1:]
    current = order[:-1][same_user]
    following = order[1:][same_user]
    next_observed[current] = True
    exits[current] = (sessions[current] != sessions[following]).float()
    return (
        torch.cat((labels, exits[:, None]), dim=1),
        torch.cat((masks, next_observed.float()[:, None]), dim=1),
    )
