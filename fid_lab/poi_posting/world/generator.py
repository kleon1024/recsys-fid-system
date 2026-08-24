"""GPU teacher-hidden creator, POI catalog, retrieval, and response world."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from ...simulation.randomness import normal as counter_normal
from ...simulation.randomness import uniform as counter_uniform
from ...simulation.randomness import uniform_for_items
from .contracts import POSTING_ROUTES, PostingWorldConfig
from .teacher import NeuralSupplyTeacher, build_neural_supply_teacher


FEATURE_NAMES = (
    "draft_similarity", "history_similarity", "same_city", "same_category",
    "popularity", "quality", "commerce", "distance", "creator_activity",
    "preference_proxy", "permission_precise", "permission_coarse",
    "route_semantic", "route_history",
)


@dataclass(frozen=True)
class PostingCatalog:
    city: torch.Tensor
    category: torch.Tensor
    semantic: torch.Tensor
    coordinates: torch.Tensor
    popularity: torch.Tensor
    quality: torch.Tensor
    commerce: torch.Tensor


@dataclass(frozen=True)
class PostingRequests:
    request_id: torch.Tensor
    latent_draft: torch.Tensor
    observed_draft: torch.Tensor
    history: torch.Tensor
    latent_city: torch.Tensor
    observed_city: torch.Tensor
    latent_category: torch.Tensor
    observed_category: torch.Tensor
    location: torch.Tensor
    permission: torch.Tensor
    creator_activity: torch.Tensor
    preference_proxy: torch.Tensor
    outside_preference: torch.Tensor
    creator_id: torch.Tensor
    request_step: torch.Tensor


@dataclass(frozen=True)
class PostingWorld:
    config: PostingWorldConfig
    catalog: PostingCatalog
    requests: PostingRequests
    category_basis: torch.Tensor
    teacher: NeuralSupplyTeacher | None


@dataclass(frozen=True)
class CandidateSet:
    item_ids: torch.Tensor
    route_bits: torch.Tensor
    recall_scores: torch.Tensor
    audit_oracle_item: torch.Tensor
    audit_oracle_recalled: torch.Tensor


def _normalize(values):
    return functional.normalize(values, dim=-1)


def _uniform(request_id, item_id, stream, seed):
    return uniform_for_items(request_id, item_id, 0, stream, seed)


def _gumbel(request_id, item_id, stream, seed):
    uniform = _uniform(request_id, item_id, stream, seed).clamp(1e-7, 1 - 1e-7)
    return -torch.log(-torch.log(uniform))


def _build_catalog(config, generator, device, category_basis):
    item_id = torch.arange(config.items, device=device)
    cell = torch.div(item_id, config.items_per_cell, rounding_mode="floor")
    city = torch.div(cell, config.categories, rounding_mode="floor")
    category = torch.remainder(cell, config.categories)
    semantic = _normalize(
        category_basis[category]
        + 0.38 * torch.randn(
            config.items, config.semantic_dim, generator=generator, device=device
        )
    )
    city_centers = torch.randn(
        config.cities, 2, generator=generator, device=device
    ) * 2.5
    coordinates = city_centers[city] + 0.35 * torch.randn(
        config.items, 2, generator=generator, device=device
    )
    popularity = torch.sigmoid(
        1.1 * torch.randn(config.items, generator=generator, device=device) - 0.4
    )
    quality = torch.sigmoid(
        torch.randn(config.items, generator=generator, device=device)
    )
    commerce = torch.sigmoid(
        0.7 * quality
        + 0.8 * torch.randn(config.items, generator=generator, device=device)
    )
    return (
        PostingCatalog(
            city, category, semantic, coordinates, popularity, quality, commerce
        ),
        city_centers,
    )


def _build_requests(config, generator, device, category_basis, city_centers):
    request_id = torch.arange(config.requests, device=device)
    latent_city = torch.randint(
        config.cities, (config.requests,), generator=generator, device=device
    )
    latent_category = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    secondary = torch.randint(
        config.categories, (config.requests,), generator=generator, device=device
    )
    latent_draft = _normalize(
        category_basis[latent_category]
        + 0.28 * category_basis[secondary]
        + 0.32 * torch.randn(
            config.requests, config.semantic_dim, generator=generator, device=device
        )
    )
    observed_draft = _normalize(
        latent_draft
        + 0.42 * torch.randn(
            config.requests, config.semantic_dim, generator=generator, device=device
        )
    )
    history_category = torch.where(
        torch.rand(config.requests, generator=generator, device=device) < 0.72,
        latent_category,
        secondary,
    )
    history = _normalize(
        category_basis[history_category]
        + 0.48 * torch.randn(
            config.requests, config.semantic_dim, generator=generator, device=device
        )
    )
    permission = torch.multinomial(
        torch.tensor([0.56, 0.29, 0.15], device=device),
        config.requests,
        replacement=True,
        generator=generator,
    )
    wrong_city = torch.randint(
        config.cities, (config.requests,), generator=generator, device=device
    )
    ip_wrong = (
        (permission == 2)
        & (torch.rand(config.requests, generator=generator, device=device) < 0.12)
    )
    observed_city = torch.where(ip_wrong, wrong_city, latent_city)
    location_noise = torch.where(
        permission[:, None] == 0,
        torch.full((config.requests, 2), 0.08, device=device),
        torch.where(
            permission[:, None] == 1,
            torch.full((config.requests, 2), 0.45, device=device),
            torch.full((config.requests, 2), 0.90, device=device),
        ),
    )
    location = city_centers[observed_city] + location_noise * torch.randn(
        config.requests, 2, generator=generator, device=device
    )
    creator_activity = torch.sigmoid(
        torch.randn(config.requests, generator=generator, device=device)
    )
    preference_proxy = torch.clamp(
        creator_activity
        + 0.30 * torch.randn(config.requests, generator=generator, device=device),
        0.0,
        1.0,
    )
    outside_preference = (
        0.35
        - 0.75 * creator_activity
        + 0.35 * torch.randn(
            config.requests, generator=generator, device=device
        )
    )
    observed_category = (observed_draft @ category_basis.T).argmax(dim=1)
    return PostingRequests(
        request_id, latent_draft, observed_draft, history, latent_city,
        observed_city, latent_category, observed_category, location,
        permission, creator_activity, preference_proxy, outside_preference,
        request_id, torch.zeros_like(request_id),
    )


def _build_creator_requests(
    config, device, category_basis, city_centers,
    request_start=0, request_count=None,
):
    count = config.requests if request_count is None else request_count
    request_id = torch.arange(
        request_start, request_start + count, device=device
    )
    creator_id = torch.remainder(request_id, config.creators)
    request_step = torch.div(request_id, config.creators, rounding_mode="floor")
    all_creators = torch.arange(config.creators, device=device)
    primary = torch.floor(
        counter_uniform(all_creators, 0, 210, config.seed) * config.categories
    ).long()
    secondary = torch.floor(
        counter_uniform(all_creators, 0, 212, config.seed) * config.categories
    ).long()
    city = torch.floor(
        counter_uniform(all_creators, 0, 214, config.seed) * config.cities
    ).long()
    activity_trait = counter_normal(all_creators, 0, 216, config.seed)
    activity_logit = (
        activity_trait[creator_id]
        + 0.18 * torch.sin(request_step.float() * 0.9)
        + 0.18 * counter_normal(request_id, 0, 218, config.seed)
    )
    activity = torch.sigmoid(activity_logit.double()).float()
    drift = torch.sigmoid((request_step.float() - 3.0) * 0.8)[:, None]
    latent_draft = _normalize(
        category_basis[primary[creator_id]]
        + (0.18 + 0.32 * drift) * category_basis[secondary[creator_id]]
        + 0.28 * counter_normal(
            request_id, 0, 220, config.seed, config.semantic_dim
        )
    )
    observed_draft = _normalize(
        latent_draft + 0.45 * counter_normal(
            request_id, 0, 222, config.seed, config.semantic_dim
        )
    )
    history = _normalize(
        category_basis[primary[creator_id]] + 0.35 * category_basis[secondary[creator_id]]
        + 0.35 * counter_normal(
            request_id, 0, 224, config.seed, config.semantic_dim
        )
    )
    permission_draw = counter_uniform(all_creators, 0, 226, config.seed)
    permission_by_creator = torch.where(
        permission_draw < 0.56, 0,
        torch.where(permission_draw < 0.85, 1, 2),
    )
    permission = permission_by_creator[creator_id]
    latent_city = city[creator_id]
    wrong_city = torch.floor(
        counter_uniform(request_id, 0, 228, config.seed) * config.cities
    ).long()
    ip_wrong = (permission == 2) & (
        counter_uniform(request_id, 0, 230, config.seed) < 0.12
    )
    observed_city = torch.where(ip_wrong, wrong_city, latent_city)
    noise = torch.where(
        permission[:, None] == 0, 0.08,
        torch.where(permission[:, None] == 1, 0.45, 0.90),
    )
    location = city_centers[observed_city] + noise * counter_normal(
        request_id, 0, 232, config.seed, 2
    )
    preference_proxy = torch.clamp(
        activity + 0.25 * counter_normal(request_id, 0, 234, config.seed),
        0.0, 1.0,
    )
    outside = 0.30 - 0.72 * activity + 0.10 * request_step.float() + (
        0.30 * counter_normal(request_id, 0, 236, config.seed)
    )
    observed_category = (observed_draft @ category_basis.T).argmax(dim=1)
    return PostingRequests(
        request_id, latent_draft, observed_draft, history, latent_city,
        observed_city, primary[creator_id], observed_category, location,
        permission, activity, preference_proxy, outside, creator_id, request_step,
    )


def build_world(config: PostingWorldConfig):
    device = torch.device(config.device)
    if config.world_version == "creator-neural-supply-v4":
        catalog_seed = config.seed if config.catalog_seed is None else config.catalog_seed
        catalog_generator = torch.Generator(device=device).manual_seed(catalog_seed)
        category_basis = _normalize(torch.randn(
            config.categories, config.semantic_dim,
            generator=catalog_generator, device=device,
        ))
        catalog, city_centers = _build_catalog(
            config, catalog_generator, device, category_basis
        )
        requests = _build_creator_requests(
            config, device, category_basis, city_centers
        )
        return PostingWorld(
            config, catalog, requests, category_basis,
            build_neural_supply_teacher(device),
        )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    category_basis = _normalize(torch.randn(
        config.categories, config.semantic_dim, generator=generator, device=device
    ))
    catalog, city_centers = _build_catalog(
        config, generator, device, category_basis
    )
    requests = _build_requests(
        config, generator, device, category_basis, city_centers
    )
    return PostingWorld(
        config, catalog, requests, category_basis, None
    )


def build_world_partition(config: PostingWorldConfig, request_start, request_count):
    """Build an exact request slice of Supply V4 without changing its random world."""
    if config.world_version != "creator-neural-supply-v4":
        raise ValueError("partitioned worlds require creator-neural-supply-v4")
    if request_start < 0 or request_count < 1:
        raise ValueError("invalid Supply V4 request partition")
    if request_start + request_count > config.requests:
        raise ValueError("Supply V4 request partition exceeds the configured world")
    device = torch.device(config.device)
    catalog_seed = config.seed if config.catalog_seed is None else config.catalog_seed
    generator = torch.Generator(device=device).manual_seed(catalog_seed)
    category_basis = _normalize(torch.randn(
        config.categories, config.semantic_dim,
        generator=generator, device=device,
    ))
    catalog, city_centers = _build_catalog(
        config, generator, device, category_basis
    )
    requests = _build_creator_requests(
        config, device, category_basis, city_centers,
        request_start, request_count,
    )
    return PostingWorld(
        config, catalog, requests, category_basis,
        build_neural_supply_teacher(device),
    )


def _cell_item(config, city, category, offset):
    cell = city * config.categories + category
    return cell * config.items_per_cell + torch.remainder(
        offset, config.items_per_cell
    )


def _route_items(world: PostingWorld):
    config, requests = world.config, world.requests
    rank = torch.arange(config.route_candidates, device=requests.request_id.device)
    request = requests.request_id[:, None]
    popular_category = torch.remainder(rank[None, :] * 5 + 1, config.categories)
    popular = _cell_item(
        config,
        requests.observed_city[:, None],
        popular_category,
        rank[None, :],
    )
    geo_category = torch.remainder(
        request * 13 + rank[None, :] * 7, config.categories
    )
    geo = _cell_item(
        config,
        requests.observed_city[:, None],
        geo_category,
        request * 17 + rank[None, :] * 11,
    )
    semantic = _cell_item(
        config,
        requests.observed_city[:, None],
        requests.observed_category[:, None],
        request * 19 + rank[None, :] * 13,
    )
    history_category = (requests.history @ world.category_basis.T).argmax(dim=1)
    history = _cell_item(
        config,
        requests.observed_city[:, None],
        history_category[:, None],
        request * 23 + rank[None, :] * 17,
    )
    return torch.stack((popular, geo, semantic, history), dim=1)


def hidden_utility(world: PostingWorld, item_ids):
    if world.teacher is not None:
        return hidden_supply_outputs(world, item_ids)[:, :, 0]
    catalog, requests = world.catalog, world.requests
    latent_similarity = torch.einsum(
        "bkd,bd->bk", catalog.semantic[item_ids], requests.latent_draft
    )
    history_similarity = torch.einsum(
        "bkd,bd->bk", catalog.semantic[item_ids], requests.history
    )
    same_city = (catalog.city[item_ids] == requests.latent_city[:, None]).float()
    same_category = (
        catalog.category[item_ids] == requests.latent_category[:, None]
    ).float()
    nonlinear = torch.sin(
        3.0 * latent_similarity + 1.7 * catalog.quality[item_ids]
    )
    return (
        1.65 * latent_similarity
        + 0.40 * history_similarity
        + 0.52 * same_city
        + 0.35 * same_category
        + 0.25 * catalog.quality[item_ids]
        + 0.10 * catalog.popularity[item_ids]
        + 0.16 * nonlinear
    )


def hidden_supply_outputs(world: PostingWorld, item_ids):
    catalog, requests = world.catalog, world.requests
    latent_similarity = torch.einsum(
        "bkd,bd->bk", catalog.semantic[item_ids], requests.latent_draft
    )
    history_similarity = torch.einsum(
        "bkd,bd->bk", catalog.semantic[item_ids], requests.history
    )
    inputs = torch.stack((
        latent_similarity,
        history_similarity,
        (catalog.city[item_ids] == requests.latent_city[:, None]).float(),
        (catalog.category[item_ids] == requests.latent_category[:, None]).float(),
        catalog.quality[item_ids], catalog.popularity[item_ids],
        catalog.commerce[item_ids],
        requests.creator_activity[:, None].expand_as(latent_similarity),
        requests.outside_preference[:, None].expand_as(latent_similarity),
        requests.request_step[:, None].float().expand_as(latent_similarity) / 8.0,
        (requests.permission[:, None] == 0).float().expand_as(latent_similarity),
        (requests.permission[:, None] == 1).float().expand_as(latent_similarity),
    ), dim=2)
    return world.teacher(inputs)


def retrieve(world: PostingWorld, enabled_routes):
    unknown = set(enabled_routes) - set(POSTING_ROUTES)
    if unknown:
        raise ValueError(f"unsupported posting routes: {sorted(unknown)}")
    route_items = _route_items(world)
    ranks = torch.arange(
        1, world.config.route_candidates + 1,
        device=route_items.device,
    ).float()
    weights = torch.tensor([0.80, 0.90, 1.00, 0.95], device=route_items.device)
    score = weights[None, :, None] / (20.0 + ranks[None, None, :])
    valid = torch.tensor(
        [name in enabled_routes for name in POSTING_ROUTES],
        device=route_items.device,
    )[None, :, None]
    score = score.expand(len(route_items), -1, -1).masked_fill(~valid, -1e9)
    flat_items = route_items.flatten(1)
    flat_scores = score.flatten(1)
    order = torch.argsort(flat_scores, dim=1, descending=True, stable=True)
    sorted_items = flat_items.gather(1, order)
    sorted_scores = flat_scores.gather(1, order)
    duplicate = torch.zeros_like(sorted_items, dtype=torch.bool)
    for position in range(1, sorted_items.shape[1]):
        duplicate[:, position] = (
            sorted_items[:, :position] == sorted_items[:, position : position + 1]
        ).any(dim=1)
    sorted_scores = sorted_scores.masked_fill(duplicate, -1e9)
    positions = torch.topk(
        sorted_scores, world.config.merged_candidates, dim=1
    ).indices
    item_ids = sorted_items.gather(1, positions)
    recall_scores = sorted_scores.gather(1, positions)
    route_bits = torch.zeros_like(item_ids)
    for route_index in range(len(POSTING_ROUTES)):
        hit = (
            item_ids[:, :, None] == route_items[:, route_index, None, :]
        ).any(dim=2)
        route_bits |= hit.long() << route_index
    audit_rank = torch.arange(32, device=item_ids.device)[None, :]
    audit = _cell_item(
        world.config,
        world.requests.latent_city[:, None],
        world.requests.latent_category[:, None],
        world.requests.request_id[:, None] * 29 + audit_rank * 19,
    )
    audit_score = hidden_utility(world, audit)
    audit_oracle = audit.gather(1, audit_score.argmax(dim=1, keepdim=True)).squeeze(1)
    recalled = (item_ids == audit_oracle[:, None]).any(dim=1)
    return CandidateSet(
        item_ids, route_bits, recall_scores, audit_oracle, recalled
    )


def candidate_features(world: PostingWorld, candidates: CandidateSet):
    catalog, requests = world.catalog, world.requests
    item_ids = candidates.item_ids
    draft_similarity = torch.einsum(
        "bkd,bd->bk", catalog.semantic[item_ids], requests.observed_draft
    )
    history_similarity = torch.einsum(
        "bkd,bd->bk", catalog.semantic[item_ids], requests.history
    )
    distance = torch.linalg.vector_norm(
        catalog.coordinates[item_ids] - requests.location[:, None, :], dim=2
    )
    return torch.stack((
        draft_similarity,
        history_similarity,
        (catalog.city[item_ids] == requests.observed_city[:, None]).float(),
        (catalog.category[item_ids] == requests.observed_category[:, None]).float(),
        catalog.popularity[item_ids],
        catalog.quality[item_ids],
        catalog.commerce[item_ids],
        torch.log1p(distance),
        requests.creator_activity[:, None].expand_as(draft_similarity),
        requests.preference_proxy[:, None].expand_as(draft_similarity),
        (requests.permission[:, None] == 0).float().expand_as(draft_similarity),
        (requests.permission[:, None] == 1).float().expand_as(draft_similarity),
        ((candidates.route_bits & (1 << 2)) > 0).float(),
        ((candidates.route_bits & (1 << 3)) > 0).float(),
    ), dim=2)


def rule_score(features):
    return (
        0.45 * features[:, :, 2]
        - 0.35 * features[:, :, 7]
        + 0.30 * features[:, :, 4]
        + 0.18 * features[:, :, 5]
    )


def simulate_response(world: PostingWorld, candidates: CandidateSet, scores):
    top = torch.topk(scores, world.config.exposed_candidates, dim=1).indices
    exposed_items = candidates.item_ids.gather(1, top)
    hidden_outputs = (
        hidden_supply_outputs(world, exposed_items)
        if world.teacher is not None else None
    )
    utility = (
        hidden_outputs[:, :, 0]
        if hidden_outputs is not None else hidden_utility(world, exposed_items)
    )
    position = torch.arange(
        world.config.exposed_candidates, device=scores.device
    ).float()
    utility = utility - 0.10 * position[None, :]
    request = world.requests.request_id[:, None]
    choice_value = utility + _gumbel(
        request, exposed_items, 101, world.config.seed
    )
    outside = world.requests.outside_preference + _gumbel(
        world.requests.request_id,
        torch.full_like(world.requests.request_id, world.config.items + 1),
        102,
        world.config.seed,
    )
    best_value, selected_rank = choice_value.max(dim=1)
    selected = best_value > outside
    selected_item = exposed_items.gather(
        1, selected_rank[:, None]
    ).squeeze(1)
    selected_utility = utility.gather(1, selected_rank[:, None]).squeeze(1)
    selected_similarity = (
        world.catalog.semantic[selected_item] * world.requests.latent_draft
    ).sum(dim=1)
    selected_hidden = (
        hidden_outputs.gather(
            1, selected_rank[:, None, None].expand(-1, 1, hidden_outputs.shape[2])
        ).squeeze(1)
        if hidden_outputs is not None else None
    )
    publish_probability = (
        torch.sigmoid(selected_hidden[:, 1])
        if selected_hidden is not None else torch.sigmoid(
            -1.45 + 1.05 * selected_utility
            + 0.65 * world.requests.creator_activity
            + 0.35 * world.catalog.quality[selected_item]
            - 0.25 * world.catalog.popularity[selected_item].square()
        )
    )
    publish_draw = _uniform(
        world.requests.request_id, selected_item, 103, world.config.seed
    )
    published = selected & (publish_draw < publish_probability)
    relevance = (
        torch.sigmoid(selected_hidden[:, 2]) if selected_hidden is not None
        else torch.sigmoid(3.5 * (selected_similarity - 0.25))
    )
    supply_quality = (
        torch.sigmoid(selected_hidden[:, 3]) if selected_hidden is not None
        else torch.sigmoid(
            1.4 * selected_similarity
            + 0.9 * world.catalog.quality[selected_item]
            - 0.4 * world.catalog.popularity[selected_item]
        )
    )
    downstream = (
        torch.nn.functional.softplus(selected_hidden[:, 5])
        if selected_hidden is not None else torch.zeros_like(relevance)
    )
    feed_stay_seconds = published.float() * (
        0.80 + 1.60 * supply_quality + 0.90 * relevance + 0.35 * downstream
    )
    feed_active_day = published.float() * torch.clamp(
        0.0015 * supply_quality + 0.0008 * relevance, max=0.003
    )
    negative_risk = (
        torch.sigmoid(selected_hidden[:, 4]) if selected_hidden is not None
        else torch.sigmoid(
            -4.0 - 1.4 * selected_similarity
            + 0.8 * world.catalog.popularity[selected_item]
        )
    )
    negative = published.float() * negative_risk
    labels = torch.zeros(
        len(scores), scores.shape[1], 3, device=scores.device
    )
    batch = torch.arange(len(scores), device=scores.device)
    selected_candidate_index = top.gather(
        1, selected_rank[:, None]
    ).squeeze(1)
    labels[batch[selected], selected_candidate_index[selected], 0] = 1.0
    labels[batch[published], selected_candidate_index[published], 1] = 1.0
    label_masks = torch.ones_like(labels)
    label_masks[:, :, 2] = 0.0
    labels[batch[published], selected_candidate_index[published], 2] = (
        relevance[published] > 0.555
    ).float()
    label_masks[batch[published], selected_candidate_index[published], 2] = 1.0
    return {
        "top_indices": top,
        "selected": selected,
        "published": published,
        "selected_item": selected_item,
        "selected_relevance": relevance,
        "supply_quality": supply_quality,
        "feed_stay_seconds": feed_stay_seconds,
        "feed_active_day": feed_active_day,
        "negative": negative,
        "selected_content_negative_risk": negative_risk,
        "labels": labels,
        "label_masks": label_masks,
    }
