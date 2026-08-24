"""Shared routing, recall, coarse, fine, and mixed-slate construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....scoring import request_standardize
from ...randomness.counter import uniform, uniform_for_items
from ..contracts import (
    SURFACE_CONTRACTS,
    ItemKind,
    Surface,
    TwinConfig,
    TwinPolicy,
)
from ..ledger import candidate_history_signals, within_request_unique
from ..platform.fids import TwinFidEncoder
from ..platform.state import CatalogState, UserState
from ..world.context import ContextState
from .models import CandidateScoringContext, ServingStack, as_serving_stack


CANDIDATE_FEATURES = (
    "observed_affinity", "realtime_affinity", "quality", "freshness",
    "popularity", "same_country", "same_region", "trend", "query_match",
    "price_match", "affordability", "merchant_quality", "inventory",
    "live_active", "open_now", "ad_pacing", "ad_ecpm_prior", "risk",
    "author_fatigue", "cluster_fatigue", "topic_fatigue", "kind_fatigue",
    "repeated_item", "user_lifecycle", "user_activity_tier",
    "user_socioeconomic", "spending_power_estimate",
    "satisfaction_estimate", "fatigue_counter", "user_cold_start",
    "trend_affinity_estimate", "commerce_intent_estimate",
    "local_intent_estimate", "creator_intent_estimate",
    "user_query_strength", "session_depth", "local_hour_sin",
    "local_hour_cos", "surface_intent",
    *(f"kind:{kind.name.lower()}" for kind in ItemKind),
    *(f"route:{index}" for index in range(6)),
)


@dataclass
class CandidateBatch:
    item_ids: torch.Tensor
    item_kind: torch.Tensor
    route: torch.Tensor
    recall_score: torch.Tensor
    coarse_score: torch.Tensor
    fine_score: torch.Tensor
    eligible: torch.Tensor
    exposed_item_ids: torch.Tensor
    exposed_scores: torch.Tensor
    exposed_propensity: torch.Tensor
    history_signals: dict[str, torch.Tensor]
    feature_values: torch.Tensor
    sparse_fids: torch.Tensor
    sparse_buckets: torch.Tensor


def _allowed_kind_table(device):
    width = max(len(contract.allowed_kinds) for contract in SURFACE_CONTRACTS.values())
    table = torch.zeros(len(Surface), width, device=device, dtype=torch.long)
    lengths = torch.zeros(len(Surface), device=device, dtype=torch.long)
    for surface, contract in SURFACE_CONTRACTS.items():
        values = torch.tensor(
            [int(kind) for kind in contract.allowed_kinds], device=device
        )
        table[int(surface), : len(values)] = values
        table[int(surface), len(values) :] = values[-1]
        lengths[int(surface)] = len(values)
    return table, lengths


def _candidate_ids(
    config: TwinConfig, users: UserState, surface: torch.Tensor, step: int,
):
    width = config.routes * config.route_candidates
    position = torch.arange(width, device=users.user_id.device)
    route = torch.div(position, config.route_candidates, rounding_mode="floor")
    table, lengths = _allowed_kind_table(users.user_id.device)
    kind_position = torch.remainder(
        position[None, :] + users.user_id[:, None], lengths[surface, None]
    )
    desired_kind = table[surface[:, None], kind_position]
    base = torch.remainder(
        users.user_id[:, None] * 1_103_515_245
        + position[None, :] * 12_345
        + step * 48_271
        + route[None, :] * 104_729
        + config.seed * 503,
        config.catalog_items,
    )
    item = torch.div(base, len(ItemKind), rounding_mode="floor") * len(ItemKind)
    item = item + desired_kind
    item = torch.remainder(item, config.catalog_items)
    return item.long(), route[None, :].expand(len(users.user_id), -1)


def _user_context_features(users, surface, local_hour, shape):
    def broadcast(value):
        return value[:, None].expand(shape)

    return {
        "user_lifecycle": broadcast(users.lifecycle.float() / 3.0),
        "user_activity_tier": broadcast(users.activity_tier.float() / 3.0),
        "user_socioeconomic": broadcast(users.socioeconomic.float() / 4.0),
        "spending_power_estimate": broadcast(
            users.spending_power_estimate
        ),
        "satisfaction_estimate": broadcast(
            users.satisfaction_estimate
        ),
        "fatigue_counter": broadcast(users.fatigue_counter),
        "user_cold_start": broadcast(users.cold_start_confidence),
        "trend_affinity_estimate": broadcast(
            users.trend_affinity_estimate
        ),
        "commerce_intent_estimate": broadcast(
            users.commerce_intent_estimate
        ),
        "local_intent_estimate": broadcast(users.local_intent_estimate),
        "creator_intent_estimate": broadcast(
            users.creator_intent_estimate
        ),
        "user_query_strength": broadcast(users.query_strength),
        "session_depth": broadcast(
            users.session_depth.float().clamp_max(20) / 20
        ),
        "local_hour_sin": torch.sin(
            2.0 * torch.pi * local_hour / 24.0
        ).expand(shape),
        "local_hour_cos": torch.cos(
            2.0 * torch.pi * local_hour / 24.0
        ).expand(shape),
        "surface_intent": broadcast(
            users.surface_affinity_estimate.gather(
                1, surface[:, None]
            ).squeeze(1)
        ),
    }


def _features(
    users: UserState, catalog: CatalogState, context: ContextState,
    item_ids: torch.Tensor, surface: torch.Tensor, step: int,
):
    embedding = catalog.topic_embedding[item_ids]
    observed = torch.einsum(
        "bkd,bd->bk", embedding, users.observed_interest
    )
    realtime = torch.einsum(
        "bkd,bd->bk", embedding, users.short_interest
    )
    same_country = (
        catalog.country[item_ids] == users.country[:, None]
    ).float()
    topic = catalog.topic[item_ids]
    country_heat = context.country_topic_heat[users.country[:, None], topic]
    global_heat = context.global_topic_heat[topic]
    trend = (
        users.trend_affinity_estimate[:, None] * country_heat
        + (1.0 - users.trend_affinity_estimate[:, None]) * global_heat
    )
    local_hour = torch.remainder(
        step + users.timezone_offset[:, None], 24
    )
    affordability = torch.exp(-(
        torch.log1p(catalog.price[item_ids])
        - users.spending_power_estimate[:, None] * 3.5
    ).abs())
    live_elapsed = torch.remainder(
        step - catalog.live_start_hour[item_ids], 24
    )
    live_active = (
        live_elapsed < catalog.live_duration_hours[item_ids]
    ).float()
    open_now = (
        (local_hour >= catalog.poi_open_hour[item_ids])
        & (local_hour <= catalog.poi_close_hour[item_ids])
    ).float()
    budget_remaining = (
        catalog.ad_budget[item_ids] - catalog.ad_spend[item_ids]
    ).clamp_min(0.0)
    values = {
        "observed_affinity": observed,
        "realtime_affinity": realtime,
        "quality": catalog.quality[item_ids],
        "freshness": catalog.freshness[item_ids],
        "popularity": catalog.popularity[item_ids],
        "same_country": same_country,
        "price_match": (
            1.0 - (catalog.price_match_prior[item_ids]
                   - users.commerce_intent_estimate[:, None]).abs()
        ).clamp_min(0.0),
        "risk": catalog.risk[item_ids],
        "sponsored_value": catalog.sponsored_value[item_ids],
        "trend": trend,
        "query_match": (
            (catalog.topic[item_ids] == users.query_topic[:, None]).float()
            * users.query_strength[:, None]
        ),
        "same_region": (
            catalog.region[item_ids] == users.region[:, None]
        ).float(),
        "affordability": affordability,
        "merchant_quality": catalog.merchant_quality[item_ids],
        "inventory": catalog.inventory[item_ids],
        "live_active": live_active,
        "open_now": open_now,
        "ad_pacing": budget_remaining / catalog.ad_budget[item_ids],
        "ad_ecpm_prior": (
            catalog.ad_bid[item_ids] * catalog.sponsored_value[item_ids]
        ),
    }
    values.update(_user_context_features(
        users, surface, local_hour, item_ids.shape
    ))
    kind = catalog.kind[item_ids]
    values.update({
        f"kind:{item_kind.name.lower()}": (kind == int(item_kind)).float()
        for item_kind in ItemKind
    })
    return values


def _route_score(features, route):
    scores = torch.stack((
        features["observed_affinity"],
        features["realtime_affinity"] + 0.60 * features["query_match"],
        features["same_region"] + 0.25 * features["same_country"]
        + 0.20 * features["quality"],
        features["freshness"],
        1.0 - features["popularity"] + 0.25 * features["quality"],
        features["popularity"] + 0.15 * features["quality"],
    ), dim=2)
    return scores.gather(2, route[:, :, None]).squeeze(2)


def _ad_eligible(
    users: UserState, catalog: CatalogState, item_ids: torch.Tensor,
    policy: TwinPolicy,
):
    ad = catalog.kind[item_ids] == int(ItemKind.AD)
    recent_ad = users.ledger.kind == int(ItemKind.AD)
    ad_count = recent_ad.sum(dim=1)
    gap = torch.full_like(ad_count, users.ledger.kind.shape[1])
    any_ad = recent_ad.any(dim=1)
    gap[any_ad] = recent_ad.float().argmax(dim=1)[any_ad]
    allowed = (
        (ad_count < policy.max_ads_per_history)
        & (gap >= policy.min_ad_gap)
    )
    return ~ad | allowed[:, None]


def _business_eligible(catalog, item_ids, features):
    kind = catalog.kind[item_ids]
    product = kind == int(ItemKind.PRODUCT)
    live = kind == int(ItemKind.LIVE_ROOM)
    ad = kind == int(ItemKind.AD)
    return (
        (~product | (features["inventory"] > 0.02))
        & (~live | features["live_active"].bool())
        & (~ad | (features["ad_pacing"] > 0.0))
    )


def _rank_scores(
    stack, policy, features, history, route, kind, feature_values, surface,
    scoring_context,
):
    route_weight = torch.tensor(
        policy.route_weights, device=route.device
    )[route]
    recall = _route_score(features, route) * route_weight
    rule_coarse = (
        policy.affinity_weight * features["observed_affinity"]
        + policy.quality_weight * features["quality"]
        + policy.freshness_weight * features["freshness"]
        + policy.popularity_weight * features["popularity"]
        + policy.trend_weight * features["trend"]
        + policy.query_weight * features["query_match"]
        + policy.merchant_weight * features["merchant_quality"]
        + policy.availability_weight * (
            features["inventory"] + features["open_now"]
            + features["live_active"]
        ) / 3.0
        + policy.geo_weight * features["same_country"]
        + policy.recall_weight * recall
        - policy.risk_penalty * features["risk"]
    )
    coarse = rule_coarse
    if stack.coarse_model is not None:
        learned_coarse = stack.coarse_model.score(
            feature_values, surface, scoring_context
        )
        coarse = rule_coarse + stack.coarse_model_weight * request_standardize(
            learned_coarse
        )
    business = (
        (kind == int(ItemKind.AD)) * policy.ad_value_weight
        * features["ad_ecpm_prior"] * features["ad_pacing"]
        + (kind == int(ItemKind.POI)) * policy.local_value_weight
        * (0.6 * features["same_region"] + 0.4 * features["open_now"])
        + (kind == int(ItemKind.LIVE_ROOM)) * policy.live_value_weight
        * features["live_active"]
        + (kind == int(ItemKind.PRODUCT)) * policy.product_value_weight
        * features["affordability"] * features["inventory"]
    )
    rule_relevance = (
        coarse
        + policy.realtime_weight * features["realtime_affinity"]
        + policy.commerce_weight * features["price_match"]
        - policy.author_fatigue_penalty * history["author_fatigue"]
        - policy.cluster_fatigue_penalty * history["cluster_fatigue"]
        - policy.topic_fatigue_penalty * history["topic_fatigue"]
    )
    relevance = rule_relevance
    if stack.fine_model is not None:
        learned_fine = stack.fine_model.score(
            feature_values, surface, scoring_context
        )
        relevance = (
            rule_relevance
            + stack.fine_model_weight * request_standardize(learned_fine)
        )
    return recall, coarse, relevance + business


def _scoring_context(users, item_ids, kind, route, surface, step):
    step_tensor = torch.full_like(users.user_id, step)
    sparse_fids, sparse_buckets = TwinFidEncoder().encode_candidates(
        user_id=users.user_id,
        item_id=item_ids,
        item_kind=kind,
        surface=surface,
        route=route,
        step=step_tensor,
    )
    return CandidateScoringContext(
        user_id=users.user_id,
        item_ids=item_ids,
        item_kinds=kind,
        route=route,
        step=step_tensor,
        sparse_fids=sparse_fids,
        sparse_buckets=sparse_buckets,
        history_item_ids=users.ledger.item,
        history_kinds=users.ledger.kind,
        history_surfaces=users.ledger.surface,
        history_steps=users.ledger.step,
    )


def build_slate(
    config: TwinConfig,
    policy: TwinPolicy | ServingStack,
    users: UserState,
    catalog: CatalogState,
    context: ContextState,
    surface: torch.Tensor,
    step: int,
) -> CandidateBatch:
    stack = as_serving_stack(policy)
    policy = stack.strategy
    item_ids, route = _candidate_ids(config, users, surface, step)
    features = _features(users, catalog, context, item_ids, surface, step)
    history = candidate_history_signals(users.ledger, catalog, item_ids, step)
    features.update({
        f"route:{index}": (route == index).float()
        for index in range(config.routes)
    })
    feature_values = torch.stack(tuple(
        history[name].float() if name in history else features[name].float()
        for name in CANDIDATE_FEATURES
    ), dim=2)
    kind = catalog.kind[item_ids]
    scoring_context = _scoring_context(
        users, item_ids, kind, route, surface, step
    )
    sparse_fids = scoring_context.sparse_fids
    sparse_buckets = scoring_context.sparse_buckets
    recall, coarse, fine = _rank_scores(
        stack, policy, features, history, route, kind, feature_values, surface,
        scoring_context,
    )
    eligible = within_request_unique(item_ids) & _ad_eligible(
        users, catalog, item_ids, policy
    ) & _business_eligible(catalog, item_ids, features)
    route_enabled = torch.tensor(
        policy.enabled_routes, device=item_ids.device
    )[route]
    eligible &= route_enabled
    if policy.recent_item_hard_filter:
        eligible &= ~history["repeated_item"]
    no_candidate = ~eligible.any(dim=1)
    if no_candidate.any():
        fallback = features["risk"].masked_fill(
            ~within_request_unique(item_ids), 2.0
        ).argmin(dim=1)
        eligible[no_candidate, fallback[no_candidate]] = True
    coarse_budget = min(policy.coarse_keep, config.coarse_keep, item_ids.shape[1])
    coarse_position = torch.topk(
        coarse.masked_fill(~eligible, -1e9), coarse_budget, dim=1
    ).indices
    coarse_items = item_ids.gather(1, coarse_position)
    coarse_fine = fine.gather(1, coarse_position)
    max_slate = max(contract.slate_size for contract in SURFACE_CONTRACTS.values())
    fine_budget = min(policy.fine_keep, config.fine_keep, coarse_budget)
    ranked = torch.topk(coarse_fine, fine_budget, dim=1).indices
    ranked_items = coarse_items.gather(1, ranked)
    ranked_scores = coarse_fine.gather(1, ranked)
    greedy_items = ranked_items
    random_priority = uniform_for_items(
        users.user_id, ranked_items, step, 229, config.seed
    )
    random_order = random_priority.argsort(dim=1, descending=True)
    random_items = ranked_items.gather(1, random_order)
    random_scores = ranked_scores.gather(1, random_order)
    explore = (
        uniform(users.user_id, step, 227, config.seed)
        < policy.exploration_rate
    )
    ranked_items = torch.where(explore[:, None], random_items, ranked_items)
    ranked_scores = torch.where(explore[:, None], random_scores, ranked_scores)
    exposed_items = ranked_items[:, :max_slate]
    exposed_scores = ranked_scores[:, :max_slate]
    surface_sizes = torch.tensor(
        [SURFACE_CONTRACTS[value].slate_size for value in Surface],
        device=surface.device,
    )
    slate_size = surface_sizes[surface]
    exposed_valid = torch.arange(max_slate, device=surface.device)[None, :] < slate_size[:, None]
    exposed_items = exposed_items.masked_fill(~exposed_valid, -1)
    exposed_scores = exposed_scores.masked_fill(~exposed_valid, -1e9)
    greedy_slate = greedy_items[:, :max_slate]
    greedy_member = (
        exposed_items[:, :, None] == greedy_slate[:, None, :]
    ).any(dim=2)
    random_inclusion = slate_size.float() / max(fine_budget, 1)
    exposed_propensity = (
        (1.0 - policy.exploration_rate) * greedy_member.float()
        + policy.exploration_rate * random_inclusion[:, None]
    ).masked_fill(~exposed_valid, 0.0)
    return CandidateBatch(
        item_ids=item_ids,
        item_kind=kind,
        route=route,
        recall_score=recall,
        coarse_score=coarse,
        fine_score=fine,
        eligible=eligible,
        exposed_item_ids=exposed_items,
        exposed_scores=exposed_scores,
        exposed_propensity=exposed_propensity,
        history_signals=history,
        feature_values=feature_values,
        sparse_fids=sparse_fids,
        sparse_buckets=sparse_buckets,
    )
