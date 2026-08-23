"""Device-resident batched Feed trajectory and candidate-graph simulator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from ...simulation.contracts import DEFAULT_SEARCH_EVENT_RATE
from ...value import BUSINESS_TREE_WEIGHTS, DEFAULT_LT_CONFIG
from .lt_exchange import (
    accumulate_lt_exchange_components,
)
from .tensor_catalog import build_tensor_catalog
from .graph.candidate import ROUTE_NAMES, build_candidate_graph
from .graph.reporting import render_report
from .graph.trace import append_trace
from .graph.random import normal, uniform, uniform_for_items
from .experiment.trigger import (
    combine_tensor_ab,
    combine_tensor_trigger_ab,
    refresh_search_state,
    trigger_mask,
)
from ..tensor_policies import (
    LOCAL_EXPANSION,
    LOCAL_INTENT_RANKER,
    LOCAL_RETARGET,
    LOCAL_SEARCH,
    LOCAL_STATIC,
    PERSONALIZED,
    PERSONALIZED_1PCT,
    POPULAR,
    TensorPolicy,
)
from ..tensor_cascade import select_candidate


DEFAULT_GPU_BATCH_USERS = 200_000


@dataclass(frozen=True)
class TensorFeedConfig:
    users: int = 1_000_000
    steps: int = 24
    candidates: int = 20
    route_candidates: int = 8
    route_oversample: int = 3
    merged_candidates: int = 48
    audit_candidates: int = 24
    candidate_graph_version: str = "multiroute-rrf-coarse-v2"
    trace_users: int = 0
    trace_requests_per_user: int = 2
    topics: int = 12
    catalog_items: int = 200_000
    batch_users: int = DEFAULT_GPU_BATCH_USERS
    seed: int = 20260823
    device: str = "cuda:0"
    count_inactive_play_bug: bool = False
    signal_version: str = "industrial-cross-sequence-v1"
    max_sessions: int = 4
    requests_per_session: int = 8
    search_event_rate: float = DEFAULT_SEARCH_EVENT_RATE
    search_ttl_requests: int = 3

    def __post_init__(self) -> None:
        if self.signal_version not in {
            "industrial-cross-sequence-v1",
            "heterogeneous-nonlinear-v2",
        }:
            raise ValueError(f"unsupported signal version: {self.signal_version}")
        if not 0.0 <= self.search_event_rate <= 1.0:
            raise ValueError("search event rate must be in [0, 1]")
        if self.search_ttl_requests < 1:
            raise ValueError("search TTL must be positive")
        if self.route_candidates < 1 or self.route_oversample < 1:
            raise ValueError("route candidate budgets must be positive")
        if self.merged_candidates < self.candidates:
            raise ValueError("merged candidates must cover the coarse output")
        if self.audit_candidates < 1:
            raise ValueError("audit candidate count must be positive")
        if self.candidate_graph_version != "multiroute-rrf-coarse-v2":
            raise ValueError("unsupported candidate graph version")
        if self.trace_users < 0 or self.trace_requests_per_user < 1:
            raise ValueError("trace sampling limits are invalid")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_response(
    user_ids, step, seed, active, fatigue, satisfaction, affinity, quality,
    duration=None, stay_adjustment=None,
):
    users = len(active)
    if duration is None:
        duration = 3.0 + 177.0 * uniform(user_ids, step, 30, seed)
    play_draw = (
        uniform(user_ids, step, 31, seed)
        < torch.sigmoid(3.0 + 0.4 * affinity - 0.6 * fatigue)
    )
    played = play_draw & active
    stay_log_mean = (
        0.45 + 1.7 * affinity + 0.55 * quality + 0.20 * satisfaction - fatigue
    )
    if stay_adjustment is not None:
        stay_log_mean += stay_adjustment
    stay = torch.minimum(
        duration,
        torch.exp(
            stay_log_mean
            + 0.65 * normal(user_ids, step, 32, seed)
        ),
    ) * played
    long_view = (stay >= torch.minimum(torch.full_like(stay, 10.0), duration)) & active
    hlt = (stay >= torch.minimum(torch.full_like(stay, 30.0), duration)) & active
    like = (
        uniform(user_ids, step, 34, seed)
        < torch.sigmoid(-4.2 + 1.8 * affinity + 0.8 * quality)
    ) & played
    negative = (
        uniform(user_ids, step, 35, seed)
        < torch.sigmoid(-5.0 - 1.7 * affinity - 0.8 * quality + 2.0 * fatigue)
    ) & active
    return stay, long_view, hlt, like, negative, played, play_draw


def _sample_local_response(
    user_ids,
    step,
    seed,
    active,
    affinity,
    is_poi,
    commerce,
    poi_quality,
    inventory,
    same_city,
    search_match,
    retarget_match,
    fulfillment,
):
    anchor = (
        uniform(user_ids, step, 40, seed)
        < torch.sigmoid(
            -5.0
            + 1.7 * affinity
            + 0.7 * same_city
            + 0.5 * commerce
            + 1.4 * search_match
            + 1.1 * retarget_match
        )
    ) & active & is_poi.bool()
    detail = (
        uniform(user_ids, step, 41, seed)
        < torch.sigmoid(-1.3 + affinity + 0.8 * poi_quality + 0.7 * search_match)
    ) & anchor
    favorite = (
        uniform(user_ids, step, 42, seed)
        < torch.sigmoid(-3.2 + 0.8 * affinity + poi_quality)
    ) & detail
    order = (
        uniform(user_ids, step, 43, seed)
        < torch.sigmoid(
            -4.5
            + 1.2 * commerce
            + 0.9 * poi_quality
            + 0.9 * retarget_match
            + 1.2 * inventory
        )
    ) & detail
    paid = order & (fulfillment == 1) & (
        uniform(user_ids, step, 44, seed) < 0.92
    )
    pixel = order & (fulfillment == 2) & (
        uniform(user_ids, step, 45, seed) < 0.35
    )
    return anchor, detail, favorite, paid, pixel


def _business_and_lt_values(
    stay,
    hlt,
    like,
    negative,
    anchor,
    detail,
    favorite,
    paid,
    pixel,
    commerce,
):
    tree = BUSINESS_TREE_WEIGHTS
    lt_rates = DEFAULT_LT_CONFIG.rates
    feed_tree = (
        hlt.float() * tree["quality_view"]
        + like.float() * tree["meaningful_engagement"]
        + negative.float() * tree["negative_feedback"]
    )
    converted = paid | pixel
    terminal_intent = torch.where(
        converted,
        torch.zeros_like(stay),
        torch.where(
            favorite,
            torch.full_like(stay, tree["poi_favorite"]),
            torch.where(
                detail,
                torch.full_like(stay, tree["poi_detail"]),
                anchor.float() * tree["anchor_click"],
            ),
        ),
    )
    transaction = (
        paid.float() * tree["closed_loop_payment"]
        + pixel.float() * tree["open_loop_conversion"]
        + converted.float() * commerce * 8.0 * tree["contribution_margin"]
    )
    local_tree = terminal_intent + transaction
    commercialization = converted.float() * commerce * 8.0
    lt_value = stay / 60.0 * lt_rates["stay_minute"].unit_value
    return lt_value, feed_tree, local_tree, commercialization


def _accumulate_cells(cell_stats, user_ids, user_metrics, cohort=None):
    bucket = torch.remainder(user_ids * 1_664_525 + 1_013_904_223, 2**31)
    assigned = bucket < 2**30
    rates = user_metrics[:, 1:].clone()
    rates[:, :16] /= user_metrics[:, :1].clamp_min(1.0)
    rates[:, 19:23] /= user_metrics[:, :1].clamp_min(1.0)
    for cell, mask in enumerate((~assigned, assigned)):
        if cohort is not None:
            mask &= cohort
        values = rates[mask]
        cell_stats[cell, :, 0] += mask.sum()
        cell_stats[cell, :, 1] += values.sum(dim=0)
        cell_stats[cell, :, 2] += values.square().sum(dim=0)


def _new_user_state(config, policy, generator, device, user_ids):
    del generator
    users = len(user_ids)
    if config.signal_version == "heterogeneous-nonlinear-v2":
        interest = (
            -torch.log(uniform(
                user_ids, 0, 1, config.seed, config.topics
            ).clamp_min(1e-7))
        ).pow(1.0 / 0.8)
        interest = torch.nn.functional.normalize(interest, dim=1)
        observed_interest = torch.clamp(
            interest
            + policy.observation_noise
            * normal(user_ids, 0, 3, config.seed, config.topics),
            min=0.0,
        )
        observed_interest = torch.nn.functional.normalize(observed_interest, dim=1)
    else:
        interest = torch.nn.functional.normalize(
            normal(user_ids, 0, 5, config.seed, config.topics), dim=1
        )
        observed_interest = torch.nn.functional.normalize(
            interest
            + policy.observation_noise
            * normal(user_ids, 0, 7, config.seed, config.topics),
            dim=1,
        )
    trigger = torch.remainder(user_ids * 1_103_515_245 + 12_345, 2**31)
    return {
        "user_ids": user_ids,
        "eligible": (trigger.float() / float(2**31) < policy.eligible_fraction).float()[:, None],
        "interest": interest,
        "observed_interest": observed_interest,
        "local_observed_interest": torch.nn.functional.normalize(
            torch.clamp(
                interest
                + policy.local_observation_noise
                * normal(user_ids, 0, 9, config.seed, config.topics),
                min=0.0,
            )
            if config.signal_version == "heterogeneous-nonlinear-v2"
            else interest
            + policy.local_observation_noise
            * normal(user_ids, 0, 11, config.seed, config.topics),
            dim=1,
        ),
        "satisfaction": torch.zeros(users, device=device),
        "fatigue": torch.zeros(users, device=device),
        "active": torch.ones(users, dtype=torch.bool, device=device),
        "sessions": torch.ones(users, device=device),
        "requests_in_session": torch.zeros(users, device=device),
        "returns": torch.zeros(users, device=device),
        "search_topic": torch.floor(
            uniform(user_ids, 0, 13, config.seed) * config.topics
        ).long(),
        "search_strength": (
            torch.zeros(users, device=device)
            if config.search_event_rate > 0.0
            else uniform(user_ids, 0, 14, config.seed)
        ),
        "search_ttl": torch.zeros(users, device=device, dtype=torch.long),
        "retarget_item": torch.full((users,), -1, device=device, dtype=torch.long),
        "city": torch.floor(uniform(user_ids, 0, 15, config.seed) * 100).long(),
        "trust": torch.remainder(user_ids * 48_271 + 17, 10_007).float() / 10_007,
        "commerce_propensity": (
            torch.remainder(user_ids * 69_697 + 29, 10_009).float() / 10_009
        ),
        "last_topic": torch.full((users,), -1, device=device, dtype=torch.long),
        "topic_counts": torch.zeros(users, config.topics, device=device),
        "ads_served": torch.zeros(users, device=device, dtype=torch.long),
        "live_served": torch.zeros(users, device=device, dtype=torch.long),
        "last_ad_step": torch.full((users,), -10_000, device=device, dtype=torch.long),
    }


def _candidate_batch(config, generator, device, state, catalog, step):
    del generator, device
    graph = build_candidate_graph(config, state, catalog, step)
    item_ids = graph["item_ids"]
    shape = item_ids.shape
    has_search = state["search_ttl"] > 0
    candidate_topic = catalog.category[item_ids]
    dynamic_inventory = catalog.inventory[item_ids] * (
        uniform_for_items(
            state["user_ids"], item_ids, step, 20, config.seed
        ) > 0.01
    )
    batch = {
        "item_ids": item_ids,
        "topics": catalog.topics[item_ids],
        "candidate_topic": candidate_topic,
        "quality": catalog.quality[item_ids],
        "freshness": torch.clamp(catalog.freshness[item_ids] - 0.01 * step, 0.0, 1.0),
        "is_poi": catalog.is_poi[item_ids],
        "commerce": catalog.commerce[item_ids],
        "poi_quality": catalog.poi_quality[item_ids],
        "inventory": dynamic_inventory.float(),
        "same_city": (catalog.city[item_ids] == state["city"][:, None]).float(),
        "search_match": (candidate_topic == state["search_topic"][:, None]).float()
        * state["search_strength"][:, None]
        * has_search[:, None],
        "retarget_match": (item_ids == state["retarget_item"][:, None]).float(),
        "fulfillment": catalog.fulfillment[item_ids],
        "content_type": catalog.content_type[item_ids],
        "ad_value": catalog.ad_value[item_ids],
        "live_value": catalog.live_value[item_ids],
        "popularity": catalog.popularity[item_ids],
        "duration": catalog.duration_seconds[item_ids],
        "author": catalog.author[item_ids],
    }
    batch.update(graph)
    return batch


def _sample_step(config, policy, generator, device, state, selected, step):
    del generator, device
    affinity = (selected["topics"] * state["interest"]).sum(dim=1)
    is_live = selected["content_type"] == 1 if policy.multi_queue else torch.zeros_like(affinity).bool()
    is_ad = selected["content_type"] == 2 if policy.multi_queue else torch.zeros_like(affinity).bool()
    response_affinity = affinity + 0.03 * is_live.float() - 0.10 * is_ad.float()
    feed = _sample_response(
        state["user_ids"],
        step,
        config.seed,
        state["active"],
        state["fatigue"],
        state["satisfaction"],
        response_affinity,
        selected["quality"],
        selected.get("duration"),
        selected.get("stay_nonlinear"),
    )
    stay, long_view, quality_view, like, negative, played, play_draw = feed
    local = _sample_local_response(
        state["user_ids"],
        step,
        config.seed,
        state["active"],
        response_affinity,
        *(selected[name] for name in (
            "is_poi", "commerce", "poi_quality", "inventory", "same_city",
            "search_match", "retarget_match", "fulfillment",
        )),
    )
    anchor, detail, favorite, paid, pixel = local
    lt_value, feed_tree, local_tree, commercialization = _business_and_lt_values(
        stay, quality_view, like, negative, anchor, detail, favorite, paid,
        pixel, selected["commerce"],
    )
    effective_ad = is_ad & played
    ad_contribution = effective_ad.float() * selected["ad_value"]
    opportunity_cost = is_ad.float() * selected["organic_opportunity_cost"]
    lt_value += (
        ad_contribution
        * DEFAULT_LT_CONFIG.rates["accepted_commercialization_unit"].unit_value
    )
    ads_live_tree = (
        ad_contribution * BUSINESS_TREE_WEIGHTS["ad_value"]
        + is_live.float()
        * selected["live_value"]
        * BUSINESS_TREE_WEIGHTS["live_value"]
    )
    played_metric = play_draw if config.count_inactive_play_bug else played
    values = {
        "stay": stay, "long_view": long_view, "quality_view": quality_view,
        "like": like, "negative": negative, "played": played_metric,
        "lt_value": lt_value,
        "feed_value_tree": feed_tree,
        "local_value_tree": local_tree,
        "accepted_commercialization": ad_contribution,
        "local_commercialization": commercialization,
        "ads_live_value_tree": ads_live_tree,
        "anchor": anchor,
        "detail": detail, "paid": paid, "pixel": pixel,
        "ad_selected": is_ad,
        "effective_ad": effective_ad,
        "ad_contribution": ad_contribution,
        "organic_opportunity_cost": opportunity_cost,
        "live_selected": is_live,
        "coarse_oracle_survives": (
            selected["coarse_oracle_survives"].float() * state["active"]
        ),
        "coarse_pass_fraction": selected["coarse_pass_fraction"] * state["active"],
        "oracle_regret": selected["oracle_regret"] * state["active"],
        "poi_candidate_fraction": (
            selected["poi_candidate_fraction"] * state["active"]
        ),
        "stage_attribution": selected["stage_attribution"],
        "route_valid_counts": selected["route_valid_counts"],
        "unique_recall_count": selected["unique_recall_count"],
    }
    return values


def _advance_state(config, policy, generator, state, selected, values, step):
    del generator
    engagement = values["long_view"].float() + values["like"].float()
    state["satisfaction"] = torch.clamp(
        0.82 * state["satisfaction"] + 0.10 * engagement - 0.24 * values["negative"].float(),
        -1.0,
        1.0,
    )
    state["fatigue"] = torch.clamp(
        0.72 * state["fatigue"] + 0.08 * values["long_view"].float(), 0.0, 1.0
    )
    update = values["long_view"].float()[:, None]
    state["interest"] = torch.nn.functional.normalize(
        state["interest"] * (1.0 - 0.10 * update) + selected["topics"] * 0.10 * update,
        dim=1,
    )
    state["observed_interest"] = torch.nn.functional.normalize(
        state["observed_interest"] * (1.0 - policy.realtime_interest_rate * update)
        + selected["topics"] * policy.realtime_interest_rate * update,
        dim=1,
    )
    state["local_observed_interest"] = torch.nn.functional.normalize(
        state["local_observed_interest"]
        * (1.0 - policy.realtime_interest_rate * update)
        + selected["topics"] * policy.realtime_interest_rate * update,
        dim=1,
    )
    state["retarget_item"] = torch.where(
        values["anchor"], selected["item_ids"], state["retarget_item"]
    )
    active_weight = state["active"].float()
    state["topic_counts"].scatter_add_(
        1, selected["candidate_topic"][:, None], active_weight[:, None]
    )
    state["last_topic"] = torch.where(
        state["active"], selected["candidate_topic"], state["last_topic"]
    )
    state["ads_served"] += values["ad_selected"] & state["active"]
    state["live_served"] += values["live_selected"] & state["active"]
    state["last_ad_step"] = torch.where(
        values["ad_selected"] & state["active"],
        torch.full_like(state["last_ad_step"], step),
        state["last_ad_step"],
    )
    state["search_strength"] *= 0.78
    if config.search_event_rate > 0.0:
        state["search_ttl"] = torch.clamp(state["search_ttl"] - 1, min=0)
        state["search_strength"] *= (state["search_ttl"] > 0).float()
    state["requests_in_session"] += state["active"]
    leave = (
        uniform(state["user_ids"], step, 50, config.seed)
        < torch.sigmoid(-3.4 - 1.2 * state["satisfaction"] + 1.7 * state["fatigue"])
    ) & state["active"]
    if config.signal_version == "heterogeneous-nonlinear-v2":
        leave |= (
            state["requests_in_session"] >= config.requests_per_session
        ) & state["active"]
    can_return = state["sessions"] < config.max_sessions
    returned = leave & (
        uniform(state["user_ids"], step, 51, config.seed)
        < torch.sigmoid(1.0 + 1.6 * state["satisfaction"] - 1.1 * state["fatigue"])
    ) & can_return
    return_value = (
        returned.float()
        * DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    )
    state["returns"] += returned
    state["sessions"] += returned
    state["requests_in_session"] = torch.where(
        returned,
        torch.zeros_like(state["requests_in_session"]),
        state["requests_in_session"],
    )
    state["ads_served"] = torch.where(
        returned, torch.zeros_like(state["ads_served"]), state["ads_served"]
    )
    state["live_served"] = torch.where(
        returned, torch.zeros_like(state["live_served"]), state["live_served"]
    )
    state["last_ad_step"] = torch.where(
        returned,
        torch.full_like(state["last_ad_step"], -10_000),
        state["last_ad_step"],
    )
    state["active"] &= ~leave | returned
    return return_value, returned


def _step_totals(config, state, values):
    return torch.stack((
        state["active"].sum(), values["stay"].sum(), values["long_view"].sum(),
        values["quality_view"].sum(), values["like"].sum(), values["negative"].sum(),
        values["played"].sum(), (values["stay"] >= 3.0).sum(),
        state["sessions"].sum() * 0.0, state["returns"].sum() * 0.0,
        values["lt_value"].sum(), values["local_value_tree"].sum(), values["anchor"].sum(),
        values["detail"].sum(), values["paid"].sum(), values["pixel"].sum(),
        values["ad_selected"].sum(), values["effective_ad"].sum(),
        values["ad_contribution"].sum(), values["organic_opportunity_cost"].sum(),
        values["feed_value_tree"].sum(), values["ads_live_value_tree"].sum(),
        values["accepted_commercialization"].sum(),
        values["local_commercialization"].sum(),
        values["coarse_oracle_survives"].sum(),
        values["coarse_pass_fraction"].sum(),
        values["oracle_regret"].sum(),
        values["poi_candidate_fraction"].sum(),
    )).to(torch.float64)


def _maybe_append_trace(
    rows, config, state, candidates, selected, values, step, offset,
    measurement_start,
):
    inside_window = step < measurement_start + config.trace_requests_per_user
    if offset == 0 and inside_window:
        append_trace(rows, config, state, candidates, selected, values, step)


@torch.inference_mode()
def _simulate_batches(
    config,
    policy,
    generator,
    device,
    catalog,
    policy_schedule,
    measurement_start_step,
    trigger_kind,
):
    totals = torch.zeros(28, dtype=torch.float64, device=device)
    cell_stats = torch.zeros(2, 25, 3, dtype=torch.float64, device=device)
    lt_exchange_stats = torch.zeros(2, 6, dtype=torch.float64, device=device)
    trigger_cell_stats = torch.zeros(2, 25, 3, dtype=torch.float64, device=device)
    trigger_users = torch.zeros((), dtype=torch.long, device=device)
    candidate_diagnostics = torch.zeros(
        5 + len(ROUTE_NAMES) + 2, dtype=torch.float64, device=device
    )
    trace_rows = []
    for offset in range(0, config.users, config.batch_users):
        users = min(config.batch_users, config.users - offset)
        user_ids = torch.arange(offset, offset + users, device=device, dtype=torch.int64)
        initial_policy = policy_schedule[0] if policy_schedule else policy
        state = _new_user_state(
            config, initial_policy, generator, device, user_ids
        )
        user_metrics = torch.zeros(users, 26, device=device)
        trigger_cohort = torch.zeros(users, dtype=torch.bool, device=device)
        for step in range(config.steps):
            active_policy = (
                policy_schedule[step] if policy_schedule is not None else policy
            )
            refresh_search_state(config, state, step)
            if trigger_kind and step == measurement_start_step:
                trigger_cohort = trigger_mask(state, trigger_kind)
                trigger_users += trigger_cohort.sum()
            candidates = _candidate_batch(config, generator, device, state, catalog, step)
            selected = select_candidate(
                active_policy, user_ids, state, candidates, device, step, config
            )
            active_before = state["active"].clone()
            values = _sample_step(
                config, active_policy, generator, device, state, selected, step
            )
            measured = step >= measurement_start_step
            if measured:
                totals += _step_totals(
                    config, {**state, "active": active_before}, values
                )
                active_attribution = values["stage_attribution"][active_before]
                candidate_diagnostics[:5] += torch.bincount(
                    active_attribution, minlength=5
                ).to(torch.float64)
                candidate_diagnostics[5 : 5 + len(ROUTE_NAMES)] += (
                    values["route_valid_counts"][active_before]
                    .sum(dim=0)
                    .to(torch.float64)
                )
                candidate_diagnostics[-2] += values["unique_recall_count"][
                    active_before
                ].sum()
                candidate_diagnostics[-1] += active_before.sum()
                _maybe_append_trace(
                    trace_rows, config, state, candidates, selected, values,
                    step, offset, measurement_start_step,
                )
                user_metrics += torch.stack((
                    active_before, values["stay"], values["long_view"], values["quality_view"],
                    values["negative"], values["lt_value"], values["local_value_tree"],
                    values["anchor"], values["paid"] | values["pixel"],
                    values["ad_selected"], values["effective_ad"],
                    values["ad_contribution"], values["organic_opportunity_cost"],
                    values["feed_value_tree"], values["ads_live_value_tree"],
                    values["accepted_commercialization"],
                    values["local_commercialization"],
                    torch.zeros_like(values["stay"]),
                    values["accepted_commercialization"],
                    values["lt_value"],
                    values["coarse_oracle_survives"],
                    values["coarse_pass_fraction"],
                    values["oracle_regret"],
                    values["poi_candidate_fraction"],
                    values["stay"]
                    / 60.0
                    * DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value,
                    torch.zeros_like(values["stay"]),
                ), dim=1).float()
            return_value, returned = _advance_state(
                config, active_policy, generator, state, selected, values, step
            )
            if measured:
                totals[8] += active_before.sum() * int(
                    step == measurement_start_step
                )
                totals[8] += returned.sum()
                totals[9] += returned.sum()
                totals[10] += return_value.sum()
                user_metrics[:, 5] += return_value
                user_metrics[:, 17] += returned
                user_metrics[:, 19] += return_value
                user_metrics[:, 25] += return_value
        _accumulate_cells(cell_stats, user_ids, user_metrics)
        if trigger_kind:
            _accumulate_cells(
                trigger_cell_stats, user_ids, user_metrics, trigger_cohort
            )
        accumulate_lt_exchange_components(
            lt_exchange_stats, user_ids, user_metrics
        )
    return (
        totals,
        cell_stats,
        lt_exchange_stats,
        trigger_cell_stats,
        trigger_users,
        candidate_diagnostics,
        trace_rows,
    )


def _prepare_run(config, policy_schedule, measurement_start_step, trigger_kind):
    if policy_schedule is not None and len(policy_schedule) != config.steps:
        raise ValueError("policy schedule must contain one policy per step")
    if not 0 <= measurement_start_step < config.steps:
        raise ValueError("measurement start must be inside the trajectory")
    if trigger_kind not in {None, "post_search", "retarget"}:
        raise ValueError(f"unsupported trigger kind: {trigger_kind}")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.current_device()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    catalog = build_tensor_catalog(config, generator, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index or 0)
    return device, generator, catalog


def run_tensor_feed(
    config: TensorFeedConfig,
    policy: TensorPolicy,
    *,
    policy_schedule: tuple[TensorPolicy, ...] | None = None,
    measurement_start_step: int = 0,
    trigger_kind: str | None = None,
) -> dict[str, object]:
    device, generator, catalog = _prepare_run(
        config, policy_schedule, measurement_start_step, trigger_kind
    )
    _sync(device)
    started = perf_counter()
    simulation = _simulate_batches(
        config,
        policy,
        generator,
        device,
        catalog,
        policy_schedule,
        measurement_start_step,
        trigger_kind,
    )
    (
        totals,
        cell_stats,
        lt_exchange_stats,
        trigger_stats,
        trigger_users,
        candidate_diagnostics,
        trace_rows,
    ) = simulation
    _sync(device)
    seconds = perf_counter() - started
    return render_report(
        config,
        policy,
        policy_schedule,
        measurement_start_step,
        totals,
        cell_stats,
        lt_exchange_stats,
        trigger_kind,
        trigger_users,
        trigger_stats,
        candidate_diagnostics,
        trace_rows,
        seconds,
        device,
    )
