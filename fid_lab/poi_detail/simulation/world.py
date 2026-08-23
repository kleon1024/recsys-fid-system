"""Upstream entry context and module-specific detail-page corpora."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from ..contracts import DETAIL_MODULES, PoiDetailConfig


@dataclass(frozen=True)
class ModuleCatalog:
    semantic: torch.Tensor
    category: torch.Tensor
    quality: torch.Tensor
    price: torch.Tensor
    availability: torch.Tensor
    trust: torch.Tensor
    informativeness: torch.Tensor
    toxicity: torch.Tensor
    freshness: torch.Tensor
    risk: torch.Tensor


@dataclass(frozen=True)
class DetailRequests:
    request_id: torch.Tensor
    user_id: torch.Tensor
    latent_intent: torch.Tensor
    observed_intent: torch.Tensor
    category: torch.Tensor
    current_poi: torch.Tensor
    entry_source: torch.Tensor
    price_preference: torch.Tensor
    transaction_propensity: torch.Tensor
    history_sequence: torch.Tensor
    history_summary: torch.Tensor
    activity: torch.Tensor
    outside_preference: torch.Tensor


@dataclass(frozen=True)
class PoiDetailWorld:
    config: PoiDetailConfig
    catalogs: tuple[ModuleCatalog, ...]
    requests: DetailRequests
    category_basis: torch.Tensor


def normalize(values):
    return functional.normalize(values, dim=-1)


def deterministic_uniform(request_id, entity_id, stream, seed):
    value = torch.remainder(
        request_id.long() * 1_103_515_245
        + entity_id.long() * 48_271 + stream * 7_919 + seed * 503,
        2**31 - 1,
    )
    return (value.float() + 0.5) / float(2**31 - 1)


def deterministic_gumbel(request_id, entity_id, stream, seed):
    draw = deterministic_uniform(
        request_id, entity_id, stream, seed
    ).clamp(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(draw))


def _catalog(config, generator, device, basis, module_index):
    entity = torch.arange(config.entities_per_module, device=device)
    category = torch.remainder(entity + module_index * 7, config.categories)
    semantic = normalize(
        basis[category] + 0.42 * torch.randn(
            config.entities_per_module, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    quality = torch.sigmoid(
        torch.randn(config.entities_per_module, generator=generator, device=device)
    )
    trust = torch.sigmoid(
        0.6 * quality + torch.randn(
            config.entities_per_module, generator=generator, device=device
        )
    )
    informativeness = torch.sigmoid(
        torch.randn(config.entities_per_module, generator=generator, device=device)
    )
    toxicity = torch.sigmoid(
        -2.4 + torch.randn(
            config.entities_per_module, generator=generator, device=device
        )
    )
    risk = torch.sigmoid(-3.0 - quality + 0.8 * toxicity)
    return ModuleCatalog(
        semantic, category, quality,
        torch.sigmoid(torch.randn(
            config.entities_per_module, generator=generator, device=device
        )),
        torch.sigmoid(torch.randn(
            config.entities_per_module, generator=generator, device=device
        )),
        trust, informativeness, toxicity,
        torch.sigmoid(torch.randn(
            config.entities_per_module, generator=generator, device=device
        )),
        risk,
    )


def _requests(config, generator, device, basis):
    request_id = torch.arange(config.requests, device=device)
    user_id = torch.remainder(request_id, config.users)
    category = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    latent = normalize(
        basis[category] + 0.30 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    observed = normalize(
        latent + 0.48 * torch.randn(
            config.requests, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    history_category = torch.where(
        torch.rand(
            config.requests, config.history_length,
            generator=generator, device=device,
        ) < 0.72,
        category[:, None],
        torch.randint(
            config.categories, (config.requests, config.history_length),
            generator=generator, device=device,
        ),
    )
    history = normalize(
        basis[history_category] + 0.45 * torch.randn(
            config.requests, config.history_length, config.semantic_dim,
            generator=generator, device=device,
        )
    )
    recency = torch.linspace(0.2, 1.0, config.history_length, device=device)
    summary = normalize((history * recency[None, :, None]).sum(1))
    activity = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    transaction = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    outside = 0.55 - 0.35 * activity + 0.45 * torch.randn(
        config.requests, generator=generator, device=device
    )
    return DetailRequests(
        request_id, user_id, latent, observed, category,
        torch.remainder(request_id * 19 + user_id * 13, config.entities_per_module),
        torch.multinomial(
            torch.tensor([0.45, 0.35, 0.20], device=device),
            config.requests, replacement=True, generator=generator,
        ),
        torch.sigmoid(torch.randn(
            config.requests, generator=generator, device=device
        )),
        transaction, history, summary, activity, outside,
    )


def build_world(config: PoiDetailConfig):
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    basis = normalize(torch.randn(
        config.categories, config.semantic_dim,
        generator=generator, device=device,
    ))
    catalogs = tuple(
        _catalog(config, generator, device, basis, index)
        for index in range(len(DETAIL_MODULES))
    )
    return PoiDetailWorld(
        config, catalogs, _requests(config, generator, device, basis), basis
    )


def hidden_utility(world, module_kind, entity_ids):
    requests = world.requests
    utility = torch.empty_like(entity_ids, dtype=torch.float)
    for kind, catalog in enumerate(world.catalogs):
        mask = module_kind == kind
        ids = entity_ids[mask]
        request_index = torch.where(mask)[0]
        intent = (
            catalog.semantic[ids] * requests.latent_intent[request_index]
        ).sum(1)
        history = (
            catalog.semantic[ids] * requests.history_summary[request_index]
        ).sum(1)
        price_match = 1.0 - (
            catalog.price[ids] - requests.price_preference[request_index]
        ).abs()
        base = 1.15 * intent + 0.35 * history + 0.35 * catalog.quality[ids]
        if kind == 0:
            value = base + 0.25 * catalog.freshness[ids]
        elif kind == 1:
            value = base + 0.45 * price_match + 0.40 * catalog.availability[ids]
        else:
            value = base + 0.55 * catalog.informativeness[ids]
            value += 0.30 * catalog.trust[ids] - 0.75 * catalog.toxicity[ids]
        utility[mask] = value
    return utility
