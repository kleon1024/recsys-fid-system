"""Observable scoring signals for creator posting recommendations."""

from __future__ import annotations

import torch

from ...catalog import PublicCatalog
from ..projection import PlatformProjectionState


def posting_route_scores(
    catalog: PublicCatalog,
    state: PlatformProjectionState,
    impression: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = catalog.quality_prior + 0.15 * torch.log1p(impression)
    topic_supply = torch.zeros(
        int(catalog.topic_id.max()) + 1,
        device=catalog.item_id.device,
    )
    topic_supply.index_add_(
        0,
        catalog.topic_id[state.item_active],
        torch.ones(int(state.item_active.sum()), device=catalog.item_id.device),
    )
    topic_density = torch.log1p(topic_supply)
    scarcity = 1.0 - topic_density[catalog.topic_id] / (
        topic_density.max().clamp_min(1.0)
    )
    diverse = (
        0.55 * catalog.quality_prior
        + 0.10 * torch.log1p(impression)
        + 0.35 * scarcity
    )
    return baseline, diverse
