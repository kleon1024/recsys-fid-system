"""Cross-route Commerce eligibility owned after route generation."""

from __future__ import annotations

import torch

from ...catalog import PublicCatalog
from ...contracts import ContentKind, PlatformRequestBatch, Surface
from ..projection import PlatformProjectionState


def inventory_eligible(
    requests: PlatformRequestBatch,
    catalog: PublicCatalog,
    state: PlatformProjectionState,
    route_item: torch.Tensor,
    minimum_inventory: float,
) -> torch.Tensor:
    safe = route_item.clamp_min(0)
    commerce = (requests.surface == int(Surface.COMMERCE))[:, None, None]
    product = catalog.content_kind[safe] == int(ContentKind.PRODUCT)
    unavailable = state.item_inventory[safe] <= minimum_inventory
    return ~(commerce & product & unavailable)
