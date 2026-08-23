"""Latent Local Search query journeys and observable POI corpus."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from ..contracts import LocalSearchConfig


@dataclass(frozen=True)
class PoiCatalog:
    category: torch.Tensor
    city: torch.Tensor
    semantic: torch.Tensor
    latitude: torch.Tensor
    longitude: torch.Tensor
    quality: torch.Tensor
    popularity: torch.Tensor
    price: torch.Tensor
    availability: torch.Tensor
    closed_loop: torch.Tensor
    risk: torch.Tensor


@dataclass(frozen=True)
class SearchRequests:
    request_id: torch.Tensor
    user_id: torch.Tensor
    latent_intent: torch.Tensor
    observed_query: torch.Tensor
    latent_category: torch.Tensor
    observed_category: torch.Tensor
    city: torch.Tensor
    latitude: torch.Tensor
    longitude: torch.Tensor
    price_preference: torch.Tensor
    history_sequence: torch.Tensor
    history_summary: torch.Tensor
    history_category: torch.Tensor
    prior_detail_poi: torch.Tensor
    retarget_eligible: torch.Tensor
    source: torch.Tensor
    urgency: torch.Tensor
    activity: torch.Tensor
    outside_preference: torch.Tensor


@dataclass(frozen=True)
class LocalSearchWorld:
    config: LocalSearchConfig
    catalog: PoiCatalog
    requests: SearchRequests
    category_basis: torch.Tensor
    city_centers: torch.Tensor


def normalize(values):
    return functional.normalize(values, dim=-1)


def deterministic_uniform(request_id, poi_id, stream, seed):
    value = torch.remainder(
        request_id.long() * 1_103_515_245
        + poi_id.long() * 48_271
        + stream * 7_919
        + seed * 503,
        2**31 - 1,
    )
    return (value.float() + 0.5) / float(2**31 - 1)


def deterministic_gumbel(request_id, poi_id, stream, seed):
    draw = deterministic_uniform(
        request_id, poi_id, stream, seed
    ).clamp(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(draw))


def category_poi(config, category, offset):
    per_category = config.pois // config.categories
    return category + torch.remainder(offset, per_category) * config.categories


def _build_catalog(config, generator, device, category_basis, city_centers):
    poi = torch.arange(config.pois, device=device)
    category = torch.remainder(poi, config.categories)
    city = torch.remainder(poi * 11 + category * 3, config.cities)
    semantic = normalize(
        category_basis[category] + 0.38 * torch.randn(
            config.pois, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    location = city_centers[city] + 0.035 * torch.randn(
        config.pois, 2, generator=generator, device=device
    )
    quality = torch.sigmoid(
        torch.randn(config.pois, generator=generator, device=device)
    )
    popularity = torch.sigmoid(
        0.7 * quality + torch.randn(config.pois, generator=generator, device=device)
    )
    price = torch.sigmoid(
        torch.randn(config.pois, generator=generator, device=device)
    )
    availability = torch.sigmoid(
        torch.randn(config.pois, generator=generator, device=device)
    )
    closed_loop = torch.remainder(poi, 5) != 0
    risk = torch.sigmoid(
        -3.4 - 1.2 * quality + 0.5 * torch.randn(
            config.pois, generator=generator, device=device
        )
    )
    return PoiCatalog(
        category, city, semantic, location[:, 0], location[:, 1], quality,
        popularity, price, availability, closed_loop, risk,
    )


def _history(config, generator, device, basis, primary, secondary):
    position = torch.arange(config.history_length, device=device)[None, :]
    explore = torch.rand(
        config.requests, config.history_length,
        generator=generator, device=device,
    ) < (0.15 + 0.25 * position / config.history_length)
    random_category = torch.randint(
        config.categories, (config.requests, config.history_length),
        generator=generator, device=device,
    )
    categories = torch.where(
        explore, random_category,
        torch.where(position % 5 == 0, secondary[:, None], primary[:, None]),
    )
    sequence = normalize(
        basis[categories] + 0.45 * torch.randn(
            config.requests, config.history_length, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    recency = torch.linspace(0.2, 1.0, config.history_length, device=device)
    summary = normalize((sequence * recency[None, :, None]).sum(1))
    return sequence, summary, categories[:, -1]


def _build_requests(config, generator, device, basis, centers):
    request_id = torch.arange(config.requests, device=device)
    user_id = torch.remainder(request_id, config.users)
    primary = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    secondary = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    latent = normalize(
        basis[primary] + 0.30 * basis[secondary] + 0.25 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    observed = normalize(
        latent + 0.52 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    noisy_category = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    observed_category = torch.where(
        torch.rand(config.requests, generator=generator, device=device) < 0.78,
        primary, noisy_category,
    )
    city = torch.randint(
        config.cities, (config.requests,), generator=generator, device=device
    )
    location = centers[city] + 0.02 * torch.randn(
        config.requests, 2, generator=generator, device=device
    )
    sequence, history_summary, history_category = _history(
        config, generator, device, basis, primary, secondary
    )
    prior_detail = category_poi(
        config, history_category, request_id * 17 + user_id * 13
    )
    retarget = torch.rand(
        config.requests, generator=generator, device=device
    ) < 0.28
    source = torch.multinomial(
        torch.tensor([0.55, 0.25, 0.20], device=device),
        config.requests, replacement=True, generator=generator,
    )
    urgency = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    activity = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    outside = 0.30 - 0.50 * urgency - 0.25 * activity + 0.45 * torch.randn(
        config.requests, generator=generator, device=device
    )
    return SearchRequests(
        request_id, user_id, latent, observed, primary, observed_category,
        city, location[:, 0], location[:, 1],
        torch.sigmoid(torch.randn(
            config.requests, generator=generator, device=device
        )),
        sequence, history_summary, history_category, prior_detail, retarget,
        source, urgency, activity, outside,
    )


def build_world(config: LocalSearchConfig):
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    basis = normalize(torch.randn(
        config.categories, config.semantic_dim,
        generator=generator, device=device,
    ))
    centers = torch.rand(
        config.cities, 2, generator=generator, device=device
    )
    return LocalSearchWorld(
        config,
        _build_catalog(config, generator, device, basis, centers),
        _build_requests(config, generator, device, basis, centers),
        basis,
        centers,
    )


def hidden_utility(world, poi_ids):
    catalog, requests = world.catalog, world.requests
    semantic = torch.einsum(
        "bkd,bd->bk", catalog.semantic[poi_ids], requests.latent_intent
    )
    history = torch.einsum(
        "bkd,bd->bk", catalog.semantic[poi_ids], requests.history_summary
    )
    distance = torch.sqrt(
        (catalog.latitude[poi_ids] - requests.latitude[:, None]).square()
        + (catalog.longitude[poi_ids] - requests.longitude[:, None]).square()
    )
    price_match = 1.0 - (
        catalog.price[poi_ids] - requests.price_preference[:, None]
    ).abs()
    city_match = (catalog.city[poi_ids] == requests.city[:, None]).float()
    nonlinear = torch.sin(2.4 * semantic + 1.2 * price_match - 2.0 * distance)
    return (
        1.35 * semantic + 0.45 * history + 0.45 * city_match
        - 1.10 * distance + 0.45 * price_match
        + 0.42 * catalog.quality[poi_ids]
        + 0.30 * catalog.availability[poi_ids]
        + 0.18 * nonlinear + 0.20 * requests.urgency[:, None]
    )
