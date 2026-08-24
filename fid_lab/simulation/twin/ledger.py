"""Cross-request exposure deduplication and fatigue authority."""

from __future__ import annotations

import torch

from .platform.state import CatalogState, ExposureLedger


def candidate_history_signals(
    ledger: ExposureLedger,
    catalog: CatalogState,
    candidate_ids: torch.Tensor,
    current_step: int,
) -> dict[str, torch.Tensor]:
    """Compute bounded multi-window exposure signals without a dense cube."""
    item = catalog.item_id[candidate_ids]
    author = catalog.author[candidate_ids]
    cluster = catalog.cluster[candidate_ids]
    topic = catalog.topic[candidate_ids]
    kind = catalog.kind[candidate_ids]
    repeated_item = torch.zeros_like(candidate_ids, dtype=torch.bool)
    author_fatigue = torch.zeros_like(candidate_ids, dtype=torch.float32)
    cluster_fatigue = torch.zeros_like(author_fatigue)
    topic_fatigue = torch.zeros_like(author_fatigue)
    kind_fatigue = torch.zeros_like(author_fatigue)
    for position in range(ledger.item.shape[1]):
        valid = ledger.step[:, position] >= 0
        age = (current_step - ledger.step[:, position]).clamp_min(0).float()
        decay = torch.exp(-age / 12.0) * valid.float()
        repeated_item |= (
            item == ledger.item[:, position, None]
        ) & valid[:, None]
        author_fatigue += (
            author == ledger.author[:, position, None]
        ).float() * decay[:, None]
        cluster_fatigue += (
            cluster == ledger.cluster[:, position, None]
        ).float() * decay[:, None]
        topic_fatigue += (
            topic == ledger.topic[:, position, None]
        ).float() * decay[:, None]
        kind_fatigue += (
            kind == ledger.kind[:, position, None]
        ).float() * decay[:, None]
    return {
        "repeated_item": repeated_item,
        "author_fatigue": author_fatigue,
        "cluster_fatigue": cluster_fatigue,
        "topic_fatigue": topic_fatigue,
        "kind_fatigue": kind_fatigue,
    }


def within_request_unique(candidate_ids: torch.Tensor) -> torch.Tensor:
    valid = torch.ones_like(candidate_ids, dtype=torch.bool)
    for position in range(1, candidate_ids.shape[1]):
        valid[:, position] &= ~(
            candidate_ids[:, :position]
            == candidate_ids[:, position : position + 1]
        ).any(dim=1)
    return valid


def append_exposure(
    ledger: ExposureLedger,
    catalog: CatalogState,
    selected_item: torch.Tensor,
    surface: torch.Tensor,
    step: int,
    active: torch.Tensor,
) -> None:
    for name in ("item", "author", "cluster", "topic", "kind", "surface", "step"):
        value = getattr(ledger, name)
        value[:, 1:] = value[:, :-1].clone()
    safe_item = selected_item.clamp_min(0)
    ledger.item[:, 0] = torch.where(active, selected_item, -1)
    ledger.author[:, 0] = torch.where(active, catalog.author[safe_item], -1)
    ledger.cluster[:, 0] = torch.where(active, catalog.cluster[safe_item], -1)
    ledger.topic[:, 0] = torch.where(active, catalog.topic[safe_item], -1)
    ledger.kind[:, 0] = torch.where(active, catalog.kind[safe_item], -1)
    ledger.surface[:, 0] = torch.where(active, surface, -1)
    ledger.step[:, 0] = torch.where(
        active, torch.full_like(selected_item, step), -1
    )
