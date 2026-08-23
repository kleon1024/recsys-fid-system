"""Bounded request-level evidence sampled from the GPU candidate graph."""

from __future__ import annotations

from hashlib import sha256
import json

import torch

from .candidate import ROUTE_NAMES


def _route_names(bitmask):
    return tuple(
        name for index, name in enumerate(ROUTE_NAMES) if bitmask & (1 << index)
    )


def _ranks(scores):
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    values = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    ranks.scatter_(1, order, values[None, :].expand_as(order))
    return ranks


def append_trace(rows, config, state, candidates, selected, values, step):
    take = min(config.trace_users, len(state["user_ids"]))
    if take == 0:
        return
    recalled = candidates["recalled_item_ids"][:take].cpu()
    route_bits = candidates["recalled_route_bits"][:take].cpu()
    recall_scores = candidates["recalled_scores"][:take].cpu()
    coarse_scores = selected["coarse_scores"][:take].cpu()
    coarse_mask = selected["coarse_mask"][:take].cpu()
    candidate_items = candidates["item_ids"][:take].cpu()
    fine_scores = selected["fine_scores"][:take].cpu()
    mix_scores = selected["mix_scores"][:take].cpu()
    exposed = selected["item_ids"][:take].cpu()
    user_ids = state["user_ids"][:take].cpu()
    audit = candidates["audit_oracle_item"][:take].cpu()
    recall_ranks = _ranks(recall_scores).cpu()
    coarse_ranks = _ranks(coarse_scores).cpu()
    fine_ranks = _ranks(fine_scores).cpu()
    mix_ranks = _ranks(mix_scores).cpu()
    labels = {
        "stay_seconds": values["stay"][:take].cpu(),
        "long_view": values["long_view"][:take].cpu(),
        "quality_long_view": values["quality_view"][:take].cpu(),
        "negative_feedback": values["negative"][:take].cpu(),
        "anchor_click": values["anchor"][:take].cpu(),
        "conversion": (values["paid"] | values["pixel"])[:take].cpu(),
    }
    for user in range(take):
        candidate_index = {
            int(item): index for index, item in enumerate(candidate_items[user])
        }
        request_id = f"gpu-u{int(user_ids[user])}-r{step}"
        for recall_index, item in enumerate(recalled[user]):
            item_id = int(item)
            index = candidate_index.get(item_id)
            coarse_pass = index is not None and bool(coarse_mask[user, index])
            is_exposed = item_id == int(exposed[user])
            rows.append({
                "request_id": request_id,
                "user_id": int(user_ids[user]),
                "request_step": step,
                "candidate_id": item_id,
                "routes": _route_names(int(route_bits[user, recall_index])),
                "recall_score": float(recall_scores[user, recall_index]),
                "recall_rank": int(recall_ranks[user, recall_index]),
                "coarse_score": float(coarse_scores[user, recall_index]),
                "coarse_rank": int(coarse_ranks[user, recall_index]),
                "coarse_pass": coarse_pass,
                "fine_score": (
                    None if not coarse_pass else float(fine_scores[user, index])
                ),
                "fine_rank": (
                    None if not coarse_pass else int(fine_ranks[user, index])
                ),
                "mix_score": (
                    None if not coarse_pass else float(mix_scores[user, index])
                ),
                "mix_rank": (
                    None if not coarse_pass else int(mix_ranks[user, index])
                ),
                "exposed": is_exposed,
                "audit_oracle": item_id == int(audit[user]),
                "mature_labels": (
                    {name: float(value[user]) for name, value in labels.items()}
                    if is_exposed else {}
                ),
            })


def render_trace(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    requests = {row["request_id"] for row in rows}
    return {
        "schema": "gpu-request-candidate-trace-v1",
        "requests": len(requests),
        "candidate_rows": len(rows),
        "sha256": sha256(payload.encode()).hexdigest(),
        "rows": rows,
    }
