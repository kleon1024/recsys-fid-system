"""Provider feedback, inventory, freshness, and new-supply transitions."""

from __future__ import annotations

import torch

from ...randomness.counter import uniform
from ..contracts import ItemKind, TwinConfig
from ..environment.latent import LatentCatalogState
from ..exchange import ObservableResponse
from ..platform.state import CatalogState


def accumulate_supply_feedback(
    catalog: CatalogState, response: ObservableResponse,
) -> None:
    item = response.selected_item[response.active]
    positive = (
        response.event("long_view") | response.event("like")
        | response.event("click") | response.event("order")
        | response.event("publish")
    )[response.active].float()
    negative = response.event("negative")[response.active].float()
    size = len(catalog.item_id)
    catalog.supply_exposure += torch.bincount(item, minlength=size).float()
    catalog.supply_positive += torch.bincount(
        item, weights=positive, minlength=size
    )
    catalog.supply_negative += torch.bincount(
        item, weights=negative, minlength=size
    )
    paid = response.event("payment")[response.active].float()
    catalog.supply_payment += torch.bincount(
        item, weights=paid, minlength=size
    )
    clicked_ad = (
        response.event("click")[response.active]
        & (catalog.kind[item] == int(ItemKind.AD))
    ).float()
    catalog.ad_spend += torch.bincount(
        item, weights=clicked_ad * catalog.ad_bid[item], minlength=size
    )


def advance_supply_day(
    config: TwinConfig,
    catalog: CatalogState,
    latent_catalog: LatentCatalogState,
    day: int,
) -> dict[str, int]:
    exposure = catalog.supply_exposure.clamp_min(1.0)
    positive_rate = catalog.supply_positive / exposure
    catalog.popularity.mul_(0.88).add_(0.12 * positive_rate)
    catalog.freshness.mul_(0.82)
    catalog.inventory.add_(
        0.08 * latent_catalog.true_quality - 0.02 * catalog.supply_payment
    ).clamp_(0.0, 1.0)

    creator_exposure = torch.zeros_like(catalog.creator_motivation)
    creator_positive = torch.zeros_like(catalog.creator_motivation)
    creator_negative = torch.zeros_like(catalog.creator_motivation)
    creator_exposure.scatter_add_(0, catalog.author, catalog.supply_exposure)
    creator_positive.scatter_add_(0, catalog.author, catalog.supply_positive)
    creator_negative.scatter_add_(0, catalog.author, catalog.supply_negative)
    creator_signal = (
        0.25 * torch.log1p(creator_exposure)
        + 0.80 * creator_positive / creator_exposure.clamp_min(1.0)
        - 1.20 * creator_negative / creator_exposure.clamp_min(1.0)
    )
    catalog.creator_motivation = (
        0.90 * catalog.creator_motivation
        + 0.10 * torch.sigmoid(creator_signal)
    ).clamp(0.0, 1.0)
    creator_id = torch.arange(
        len(catalog.creator_motivation), device=catalog.item_id.device
    )
    retain_probability = torch.sigmoid(
        2.6 + 1.4 * catalog.creator_motivation
        - 0.35 * torch.log1p(creator_negative)
    )
    catalog.creator_active &= uniform(
        creator_id, day, 331, config.seed
    ) < retain_probability
    publish_probability = torch.sigmoid(
        -2.7 + 2.4 * catalog.creator_motivation
        - 0.03 * torch.log1p(catalog.creator_posts)
    ) * catalog.creator_active
    published = uniform(
        creator_id, day, 337, config.seed
    ) < publish_probability
    publishing_creator = creator_id[published]
    slot = torch.remainder(
        publishing_creator + day * len(creator_id), config.catalog_items
    )
    catalog.author[slot] = publishing_creator
    catalog.freshness[slot] = 1.0
    catalog.popularity[slot] = 0.02
    catalog.inventory[slot] = 0.6 + 0.4 * catalog.creator_motivation[
        publishing_creator
    ]
    latent_catalog.true_quality[slot] = (
        0.45 * latent_catalog.true_quality[slot]
        + 0.55 * catalog.creator_motivation[publishing_creator]
    )
    latent_catalog.true_risk[slot] = (
        0.80 * latent_catalog.true_risk[slot]
        + 0.20 * (1.0 - latent_catalog.true_quality[slot])
    )
    quality_refresh = uniform(catalog.item_id, day, 419, config.seed) < 0.15
    catalog.quality[quality_refresh] = (
        0.85 * catalog.quality[quality_refresh]
        + 0.15 * latent_catalog.true_quality[quality_refresh]
    )
    risk_refresh = uniform(catalog.item_id, day, 421, config.seed) < 0.10
    catalog.risk[risk_refresh] = (
        0.90 * catalog.risk[risk_refresh]
        + 0.10 * latent_catalog.true_risk[risk_refresh]
    )
    catalog.creator_posts[publishing_creator] += 1.0
    catalog.supply_exposure.zero_()
    catalog.supply_positive.zero_()
    catalog.supply_negative.zero_()
    catalog.supply_payment.zero_()
    catalog.ad_spend.zero_()
    return {
        "active_creators": int(catalog.creator_active.sum()),
        "new_items": int(published.sum()),
    }
