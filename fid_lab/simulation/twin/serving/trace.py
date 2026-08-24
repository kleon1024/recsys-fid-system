"""Bounded request-level candidate and outcome audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b

import torch

from ..exchange import ObservableResponse, TASKS
from ..platform.state import UserState
from .surfaces import CANDIDATE_FEATURES, CandidateBatch


@dataclass
class RequestTrace:
    rows: list[dict[str, torch.Tensor]] = field(default_factory=list)

    def append(
        self,
        users: UserState,
        surface: torch.Tensor,
        candidates: CandidateBatch,
        response: ObservableResponse,
        step: int,
        limit: int,
        served_policy: str,
        experiment_cell: str,
    ) -> None:
        active_indices = torch.nonzero(
            response.active, as_tuple=False
        ).flatten()
        active_count = len(active_indices)
        if active_count > limit:
            user_ids = users.user_id[active_indices]
            priority = torch.bitwise_and(
                user_ids * 1_103_515_245 + step * 48_271, 0x7FFFFFFF
            )
            selected = torch.topk(
                priority, limit, largest=False
            ).indices
            indices = active_indices[selected]
        else:
            indices = active_indices
        count = len(indices)
        if count <= 0:
            return
        def frozen(value, dtype=None):
            copied = value[indices].detach().cpu().clone()
            return copied.to(dtype) if dtype is not None else copied

        def stable_id(value: str) -> int:
            return int.from_bytes(
                blake2b(value.encode(), digest_size=7).digest(), "big"
            )

        self.rows.append({
            "request_id": frozen(users.user_id * 1_000_003 + step),
            "user_id": frozen(users.user_id),
            "step": torch.full((count,), step, dtype=torch.int32),
            "surface": frozen(surface, torch.int8),
            "request_sampling_probability": torch.full(
                (count,), min(limit / max(active_count, 1), 1.0),
                dtype=torch.float32,
            ),
            "served_policy_id": torch.full(
                (count,), stable_id(served_policy), dtype=torch.long
            ),
            "experiment_cell_id": torch.full(
                (count,), stable_id(experiment_cell), dtype=torch.long
            ),
            "recalled_item_ids": frozen(candidates.item_ids, torch.int32),
            "candidate_kind": frozen(candidates.item_kind, torch.int8),
            "route": frozen(candidates.route, torch.int8),
            "recall_score": frozen(candidates.recall_score, torch.float16),
            "coarse_score": frozen(candidates.coarse_score, torch.float16),
            "fine_score": frozen(candidates.fine_score, torch.float16),
            "candidate_features": frozen(
                candidates.feature_values, torch.float16
            ),
            "candidate_sparse_fids": frozen(
                candidates.sparse_fids, torch.long
            ),
            "candidate_sparse_buckets": frozen(
                candidates.sparse_buckets, torch.int32
            ),
            "eligible": frozen(candidates.eligible),
            "exposed_item_ids": frozen(
                candidates.exposed_item_ids, torch.int32
            ),
            "exposed_propensity": frozen(
                candidates.exposed_propensity, torch.float16
            ),
            "selected_item": frozen(response.selected_item, torch.int32),
            "labels": frozen(response.task),
            "label_mask": frozen(response.task_mask),
            "active": frozen(response.active),
            "history_item": frozen(users.ledger.item, torch.int32),
            "history_kind": frozen(users.ledger.kind, torch.int8),
            "history_surface": frozen(users.ledger.surface, torch.int8),
            "history_step": frozen(users.ledger.step, torch.int32),
        })

    def manifest(self) -> dict[str, object]:
        requests = sum(len(row["request_id"]) for row in self.rows)
        return {
            "schema": "multi-surface-request-candidate-trace-v1",
            "requests": requests,
            "fields": (
                list(self.rows[0]) if self.rows else []
            ),
            "tasks": list(TASKS),
            "candidate_features": list(CANDIDATE_FEATURES),
            "candidate_closure": (
                "recall, route, coarse, fine, eligibility, exposure, label"
            ),
            "version_lineage": "served_policy_id + experiment_cell_id",
            "storage": "active_requests_only; fp16 scores/features; compact ids",
        }

    def sampled(self, maximum_requests: int, salt: int = 0) -> RequestTrace:
        """Deterministic bounded read view with exact second-stage propensity."""
        if maximum_requests <= 0:
            raise ValueError("maximum trace sample must be positive")
        counts = [len(row["request_id"]) for row in self.rows]
        total = sum(counts)
        if total <= maximum_requests:
            return RequestTrace(rows=list(self.rows))
        request_ids = torch.cat([
            row["request_id"].long() for row in self.rows
        ])
        priority = torch.bitwise_and(
            request_ids * 1_103_515_245 + salt * 48_271,
            0x7FFFFFFFFFFFFFFF,
        )
        selected = torch.topk(
            priority, maximum_requests, largest=False
        ).indices.sort().values
        probability = maximum_requests / total
        sampled_rows = []
        offset = 0
        for row, count in zip(self.rows, counts, strict=True):
            local = selected[(selected >= offset) & (selected < offset + count)]
            local = local - offset
            offset += count
            if len(local) == 0:
                continue
            copied = {
                name: value[local].clone() for name, value in row.items()
            }
            copied["request_sampling_probability"].mul_(probability)
            sampled_rows.append(copied)
        return RequestTrace(rows=sampled_rows)

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self.rows:
            return {}
        return {
            name: torch.cat([row[name] for row in self.rows])
            for name in self.rows[0]
        }

    def validate(self) -> dict[str, bool]:
        values = self.tensors()
        if not values:
            raise ValueError("request trace is empty")
        recalled = values["recalled_item_ids"]
        exposed = values["exposed_item_ids"]
        selected = values["selected_item"]
        exposed_valid = exposed >= 0
        exposure_in_recall = (
            (exposed[:, :, None] == recalled[:, None, :]).any(dim=2)
            | ~exposed_valid
        ).all()
        selected_in_exposure = (
            selected[:, None] == exposed
        ).any(dim=1).all()
        label_shape = values["labels"].shape == values["label_mask"].shape
        request_step = values["step"][:, None]
        no_future_history = (
            (values["history_step"] < request_step)
            | (values["history_step"] < 0)
        ).all()
        return {
            "exposure_in_recall": bool(exposure_in_recall),
            "selected_in_exposure": bool(selected_in_exposure),
            "label_mask_shape": bool(label_shape),
            "point_in_time_history": bool(no_future_history),
        }
