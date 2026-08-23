"""Finite device-resident catalog for repeatable large-scale Feed worlds."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorCatalog:
    topics: torch.Tensor
    category: torch.Tensor
    quality: torch.Tensor
    freshness: torch.Tensor
    is_poi: torch.Tensor
    commerce: torch.Tensor
    poi_quality: torch.Tensor
    inventory: torch.Tensor
    city: torch.Tensor
    fulfillment: torch.Tensor
    content_type: torch.Tensor
    ad_value: torch.Tensor
    live_value: torch.Tensor
    popularity: torch.Tensor
    duration_seconds: torch.Tensor
    author: torch.Tensor

    @property
    def size(self) -> int:
        return len(self.quality)


def build_tensor_catalog(config, generator, device: torch.device) -> TensorCatalog:
    item_ids = torch.arange(config.catalog_items, device=device)
    hashed = torch.remainder(item_ids * 1_103_515_245 + 12_345, 2**31).float()
    hashed /= float(2**31)
    category = torch.remainder(item_ids, config.topics)
    if config.signal_version == "heterogeneous-nonlinear-v2":
        v2 = torch.Generator(device=device).manual_seed(config.seed + 707)
        raw_topics = -torch.log(
            torch.rand(
                config.catalog_items, config.topics, generator=v2, device=device
            ).clamp_min(1e-7)
        )
        raw_topics = raw_topics.pow(1.0 / 0.7)
        raw_topics[torch.arange(config.catalog_items, device=device), category] += 1.0
        topics = torch.nn.functional.normalize(raw_topics, dim=1)
    else:
        centers = torch.nn.functional.normalize(
            torch.randn(config.topics, config.topics, generator=generator, device=device),
            dim=1,
        )
        topics = torch.nn.functional.normalize(
            centers[category]
            + 0.28
            * torch.randn(
                config.catalog_items,
                config.topics,
                generator=generator,
                device=device,
            ),
            dim=1,
        )
    is_poi = torch.rand(config.catalog_items, generator=generator, device=device) < 0.28
    fulfillment = torch.where(
        torch.rand(config.catalog_items, generator=generator, device=device) < 0.65,
        torch.ones(config.catalog_items, device=device, dtype=torch.long),
        torch.full((config.catalog_items,), 2, device=device, dtype=torch.long),
    ) * is_poi.long()
    if config.signal_version == "heterogeneous-nonlinear-v2":
        quality_generator = torch.Generator(device=device).manual_seed(config.seed + 708)
        numerator = -torch.log(
            torch.rand(
                config.catalog_items, 3, generator=quality_generator, device=device
            ).clamp_min(1e-7)
        ).sum(dim=1)
        denominator_tail = -torch.log(
            torch.rand(
                config.catalog_items, 2, generator=quality_generator, device=device
            ).clamp_min(1e-7)
        ).sum(dim=1)
        quality = numerator / (numerator + denominator_tail)
    else:
        quality = torch.rand(config.catalog_items, generator=generator, device=device)
    freshness = torch.rand(config.catalog_items, generator=generator, device=device)
    commerce = torch.rand(config.catalog_items, generator=generator, device=device)
    poi_quality = torch.rand(config.catalog_items, generator=generator, device=device)
    inventory = (
        torch.rand(config.catalog_items, generator=generator, device=device) < 0.92
    ).float()
    city = torch.randint(100, (config.catalog_items,), generator=generator, device=device)
    stream_generator = torch.Generator(device=device).manual_seed(config.seed + 909)
    stream_draw = torch.rand(
        config.catalog_items, generator=stream_generator, device=device
    )
    content_type = torch.zeros(config.catalog_items, device=device, dtype=torch.long)
    content_type[stream_draw >= 0.84] = 1
    content_type[stream_draw >= 0.94] = 2
    return TensorCatalog(
        topics=topics,
        category=category,
        quality=quality,
        freshness=freshness,
        is_poi=is_poi.float(),
        commerce=commerce,
        poi_quality=poi_quality,
        inventory=inventory,
        city=city,
        fulfillment=fulfillment,
        content_type=content_type,
        ad_value=torch.rand(
            config.catalog_items, generator=stream_generator, device=device
        ),
        live_value=torch.rand(
            config.catalog_items, generator=stream_generator, device=device
        ),
        popularity=hashed,
        duration_seconds=(
            torch.clamp(
                torch.exp(
                    3.2
                    + 0.65
                    * torch.randn(
                        config.catalog_items,
                        generator=torch.Generator(device=device).manual_seed(config.seed + 709),
                        device=device,
                    )
                ),
                5.0,
                180.0,
            )
            if config.signal_version == "heterogeneous-nonlinear-v2"
            else 3.0 + 177.0 * torch.remainder(hashed * 1.618, 1.0)
        ),
        author=torch.remainder(item_ids * 2_654_435_761, 1024),
    )
