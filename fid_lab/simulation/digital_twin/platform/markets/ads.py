"""Deterministic advertiser-capacity allocation and organic backfill."""

from __future__ import annotations

from dataclasses import replace

import torch

from ...catalog import PublicCatalog
from ...contracts import ContentKind, PlatformRequestBatch, SelectionPolicyKind
from ..projection import PlatformProjectionState


def _backfill_organic(catalog, stages, row, position):
    candidate = stages.fine_item_id[row]
    safe = candidate.clamp_min(0)
    valid = candidate >= 0
    valid &= catalog.content_kind[safe] != int(ContentKind.AD)
    valid &= ~torch.isin(candidate, stages.exposed_item_id[row])
    score = stages.fine_selected_score[row].masked_fill(~valid, -torch.inf)
    choice = int(score.argmax())
    if not torch.isfinite(score[choice]):
        stages.exposed_item_id[row, position] = -1
        stages.exposed_score[row, position] = -torch.inf
        return
    stages.exposed_item_id[row, position] = candidate[choice]
    stages.exposed_score[row, position] = score[choice]


def enforce_ad_budget(
    catalog: PublicCatalog,
    requests: PlatformRequestBatch,
    state: PlatformProjectionState,
    stages,
    assignment_probability: torch.Tensor,
):
    """Allocate one experiment cell's share of advertiser capacity."""
    stages = replace(
        stages,
        exposed_item_id=stages.exposed_item_id.clone(),
        exposed_score=stages.exposed_score.clone(),
    )
    safe = stages.exposed_item_id.clamp_min(0)
    ad = (
        (stages.exposed_item_id >= 0)
        & (catalog.content_kind[safe] == int(ContentKind.AD))
    )
    row, position = torch.where(ad)
    if not len(row):
        return stages
    advertiser = catalog.advertiser_id[
        stages.exposed_item_id[row, position]
    ]
    for advertiser_id in torch.unique(advertiser, sorted=True).tolist():
        selected = advertiser == advertiser_id
        selected_row = row[selected]
        selected_position = position[selected]
        fractions = torch.unique(assignment_probability[selected_row])
        if len(fractions) != 1:
            raise ValueError("one Ads cell has inconsistent assignment probability")
        bid = state.advertiser_bid[advertiser_id]
        total_capacity = torch.floor(
            state.advertiser_budget[advertiser_id] / bid.clamp_min(1e-12)
        ).long()
        cell_capacity = int(torch.floor(total_capacity * fractions[0]))
        order = torch.argsort(requests.request_id[selected_row], stable=True)
        reject = order[cell_capacity:]
        rejected_rows = selected_row[reject]
        rejected_positions = selected_position[reject]
        if len(rejected_rows) and (
            stages.selection_policy_kind[rejected_rows]
            != int(SelectionPolicyKind.DETERMINISTIC)
        ).any():
            raise ValueError(
                "randomized Ads exposure requires constrained propensity logging"
            )
        for rejected_row, rejected_position in zip(
            rejected_rows.tolist(), rejected_positions.tolist(), strict=True,
        ):
            _backfill_organic(
                catalog, stages, rejected_row, rejected_position,
            )
    return stages
