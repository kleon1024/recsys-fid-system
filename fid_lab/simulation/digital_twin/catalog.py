"""Observable content catalog shared as public input, never latent truth."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .semantics import CONTENT_TOPIC_RESIDUAL_WEIGHT, mix_direction

from ..randomness.counter import normal, uniform
from .contracts import ContentKind


def _categorical(
    entity_id: torch.Tensor,
    cardinality: int,
    *,
    stream: int,
    seed: int,
) -> torch.Tensor:
    """Draw an ID-independent categorical assignment from a counter stream."""
    return torch.floor(
        uniform(entity_id, 0, stream, seed) * cardinality
    ).long().clamp_max(cardinality - 1)


def _content_kind(item_id: torch.Tensor, platform_seed: int) -> torch.Tensor:
    """Use an explicit mixed-media supply distribution, not item-ID modulo."""
    draw = uniform(item_id, 0, 97, platform_seed)
    boundaries = torch.tensor(
        (0.72, 0.80, 0.85, 0.89, 0.93, 0.95, 0.97, 0.99),
        device=item_id.device,
    )
    return torch.bucketize(draw, boundaries)


@dataclass(frozen=True)
class PublicCatalog:
    item_id: torch.Tensor
    content_kind: torch.Tensor
    topic_id: torch.Tensor
    content_embedding: torch.Tensor
    creator_id: torch.Tensor
    merchant_id: torch.Tensor
    advertiser_id: torch.Tensor
    product_id: torch.Tensor
    poi_id: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    publish_time: torch.Tensor
    evergreen_eligible: torch.Tensor
    duration_seconds: torch.Tensor
    quality_prior: torch.Tensor
    price: torch.Tensor
    inventory: torch.Tensor
    active: torch.Tensor

    def __post_init__(self):
        items = len(self.item_id)
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "content_embedding":
                if value.ndim != 2 or value.shape[0] != items:
                    raise ValueError("content embedding must be [item, dim]")
            elif value.shape != (items,):
                raise ValueError(f"catalog field {field.name} is not item-aligned")
        if not torch.equal(
            self.item_id,
            torch.arange(items, device=self.item_id.device),
        ):
            raise ValueError("reference catalog item IDs must be contiguous")


def _business_anchors(
    item_id: torch.Tensor,
    content_kind: torch.Tensor,
    platform_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    product_items = item_id[content_kind == int(ContentKind.PRODUCT)]
    poi_items = item_id[content_kind == int(ContentKind.POI)]
    linkable_product = (
        (content_kind == int(ContentKind.SHORT_VIDEO))
        | (content_kind == int(ContentKind.PHOTO))
        | (content_kind == int(ContentKind.CARD))
        | (content_kind == int(ContentKind.LIVE_ROOM))
    )
    linkable_poi = (
        (content_kind == int(ContentKind.SHORT_VIDEO))
        | (content_kind == int(ContentKind.PHOTO))
        | (content_kind == int(ContentKind.CARD))
    )
    product_link = (
        (content_kind == int(ContentKind.PRODUCT))
        | (
            linkable_product
            & (uniform(item_id, 0, 151, platform_seed) < 0.14)
        )
    )
    poi_link = (
        (content_kind == int(ContentKind.POI))
        | (linkable_poi & (uniform(item_id, 0, 157, platform_seed) < 0.18))
    )
    product_id = torch.full_like(item_id, -1)
    poi_id = torch.full_like(item_id, -1)
    if len(product_items):
        product_id[product_link] = product_items[
            torch.remainder(
                item_id[product_link] * 1_009 + 17, len(product_items),
            )
        ]
    if len(poi_items):
        poi_id[poi_link] = poi_items[
            torch.remainder(item_id[poi_link] * 503 + 31, len(poi_items))
        ]
    return product_id, poi_id


def build_public_catalog(
    *,
    items: int,
    creators: int,
    merchants: int,
    topics: int,
    countries: int,
    regions_per_country: int,
    embedding_dim: int,
    platform_seed: int,
    device: str | torch.device,
    advertisers: int | None = None,
    initial_active_fraction: float = 0.92,
) -> PublicCatalog:
    advertisers = merchants if advertisers is None else advertisers
    dimensions = (
        items,
        creators,
        merchants,
        advertisers,
        topics,
        countries,
        regions_per_country,
        embedding_dim,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("catalog dimensions must be positive")
    if not 0.0 < initial_active_fraction <= 1.0:
        raise ValueError("initial active fraction must be in (0, 1]")
    device = torch.device(device)
    item_id = torch.arange(items, device=device)
    topic_id = _categorical(
        item_id, topics, stream=99, seed=platform_seed,
    )
    topic_ids = torch.arange(topics, device=device)
    prototypes = torch.nn.functional.normalize(
        normal(topic_ids, 0, 101, platform_seed, embedding_dim), dim=1
    )
    content_embedding = mix_direction(
        prototypes[topic_id],
        normal(item_id, 0, 103, platform_seed, embedding_dim),
        CONTENT_TOPIC_RESIDUAL_WEIGHT,
    )
    content_kind = _content_kind(item_id, platform_seed)
    country = _categorical(
        item_id, countries, stream=105, seed=platform_seed,
    )
    region = (
        country * regions_per_country
        + _categorical(
            item_id,
            regions_per_country,
            stream=106,
            seed=platform_seed,
        )
    )
    text_quality = uniform(item_id, 0, 107, platform_seed)
    visual_quality = uniform(item_id, 0, 109, platform_seed)
    quality_prior = torch.sigmoid(
        -1.0 + 1.25 * text_quality + 1.45 * visual_quality
        + 0.35 * normal(item_id, 0, 113, platform_seed)
    )
    active = uniform(item_id, 0, 149, platform_seed) < initial_active_fraction
    product_id, poi_id = _business_anchors(
        item_id, content_kind, platform_seed,
    )
    historical_publish_time = -torch.floor(
        720.0 * uniform(item_id, 0, 127, platform_seed)
    ).long()
    post_kind = (
        (content_kind == int(ContentKind.SHORT_VIDEO))
        | (content_kind == int(ContentKind.PHOTO))
        | (content_kind == int(ContentKind.ARTICLE))
        | (content_kind == int(ContentKind.CARD))
    )
    evergreen_eligible = (
        post_kind
        & (quality_prior >= 0.72)
        & (uniform(item_id, 0, 163, platform_seed) < 0.24)
    )
    return PublicCatalog(
        item_id=item_id,
        content_kind=content_kind,
        topic_id=topic_id,
        content_embedding=content_embedding,
        creator_id=_categorical(
            item_id, creators, stream=117, seed=platform_seed,
        ),
        merchant_id=_categorical(
            item_id, merchants, stream=119, seed=platform_seed,
        ),
        advertiser_id=_categorical(
            item_id, advertisers, stream=121, seed=platform_seed,
        ),
        product_id=product_id,
        poi_id=poi_id,
        country=country,
        region=region,
        publish_time=torch.where(
            active,
            historical_publish_time,
            torch.full_like(item_id, torch.iinfo(torch.long).max),
        ),
        evergreen_eligible=evergreen_eligible,
        duration_seconds=(
            4.0 + 176.0 * uniform(item_id, 0, 131, platform_seed).square()
        ),
        quality_prior=quality_prior,
        price=torch.exp(
            -1.5 + 5.0 * uniform(item_id, 0, 137, platform_seed)
        ),
        inventory=uniform(item_id, 0, 139, platform_seed),
        active=active,
    )
