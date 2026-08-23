"""Request-level tensors for learned DGP training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .contracts import WORLD_LABEL_COUNT


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
    candidate_utility_source: str = "synthetic_oracle"

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
            "exposed_index": self.exposed_index[indices].to(
                device, non_blocking=True
            ),
        }


def load_world_split(dataset_dir: Path, split: str, max_rows: int | None = None,
                     row_selection: str = "head") -> WorldModelSplit:
    payload = torch.load(
        dataset_dir / f"{split}.pt", map_location="cpu", weights_only=False
    )["tensors"]
    total_rows = len(payload["exposed_index"])
    rows = total_rows if max_rows is None else min(total_rows, max_rows)
    if row_selection == "head":
        row_index = torch.arange(rows)
    elif row_selection == "uniform":
        row_index = torch.arange(rows) * total_rows // rows
    else:
        raise ValueError(f"unsupported row selection: {row_selection}")
    choice = payload["exposed_index"][row_index].long()
    propensity = payload["exposure_propensity"][row_index].float().clamp_min(1e-4)
    weights = (1.0 / propensity).clamp_max(20.0)
    weights /= weights.mean()
    labels, label_masks = _with_session_exit(payload, total_rows)
    labels = labels[row_index]
    label_masks = label_masks[row_index]
    return WorldModelSplit(
        selected_features=payload["candidate_features"][row_index, choice].float(),
        slate_features=payload["candidate_features"][row_index].float(),
        sequence=payload["behavior_sequence"][row_index].float(),
        labels=labels,
        label_masks=label_masks,
        weights=weights,
        lifecycle=payload["lifecycle_bucket"][row_index].long(),
        region=payload["region_bucket"][row_index].long(),
        user_ids=payload["user_id"][row_index].long(),
        request_steps=payload["request_step"][row_index].long(),
        exposed_index=choice,
        candidate_fine_scores=payload["candidate_fine_scores"][row_index].float(),
        candidate_audit_utility=payload["candidate_audit_utility"][row_index].float(),
        candidate_utility_source=payload.get(
            "candidate_utility_source", "synthetic_oracle"
        ),
    )


def _with_session_exit(payload, rows):
    labels = payload["labels"][:rows].float()
    masks = payload["label_masks"][:rows].float()
    if labels.shape[1] >= 16:
        return _pad_world_labels(labels, masks)
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
    labels, masks = (
        torch.cat((labels, exits[:, None]), dim=1),
        torch.cat((masks, next_observed.float()[:, None]), dim=1),
    )
    return _pad_world_labels(labels, masks)


def _pad_world_labels(labels, masks):
    missing = WORLD_LABEL_COUNT - labels.shape[1]
    if missing < 0:
        raise ValueError(
            f"world labels have {labels.shape[1]} columns; expected at most "
            f"{WORLD_LABEL_COUNT}"
        )
    if not missing:
        return labels, masks
    zeros = torch.zeros((len(labels), missing), dtype=labels.dtype)
    return torch.cat((labels, zeros), dim=1), torch.cat((masks, zeros), dim=1)
