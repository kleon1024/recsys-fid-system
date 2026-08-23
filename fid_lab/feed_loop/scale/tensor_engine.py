"""Device-resident batched Feed trajectory and candidate-graph simulator."""

from __future__ import annotations

from time import perf_counter

import torch

from ...value import BUSINESS_TREE_WEIGHTS, DEFAULT_LT_CONFIG
from .artifact.features import build_tensor_features
from .calibration.nonlinear import nonlinear_stay_adjustment
from .lt_exchange import (
    accumulate_lt_exchange_components,
)
from .tensor_catalog import build_tensor_catalog
from .graph.candidate import ROUTE_NAMES, build_candidate_graph
from .graph.reporting import render_report
from .graph.trace import append_trace
from .graph.random import uniform_for_items
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
from .tensor_runtime.contracts import DEFAULT_GPU_BATCH_USERS, TensorFeedConfig
from .tensor_runtime.response import (
    business_and_lt_values,
    sample_local_response,
    sample_response,
)
from .tensor_runtime.state import advance_state, new_user_state


__all__ = [
    "DEFAULT_GPU_BATCH_USERS",
    "LOCAL_EXPANSION",
    "LOCAL_INTENT_RANKER",
    "LOCAL_RETARGET",
    "LOCAL_SEARCH",
    "LOCAL_STATIC",
    "PERSONALIZED",
    "PERSONALIZED_1PCT",
    "POPULAR",
    "TensorFeedConfig",
    "TensorPolicy",
    "combine_tensor_ab",
    "combine_tensor_trigger_ab",
    "candidate_batch",
    "new_user_state",
    "prepare_run",
    "run_tensor_feed",
    "sample_step",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def _accumulate_cells(cell_stats, user_ids, user_metrics, cohort=None):
    bucket = torch.remainder(user_ids * 1_664_525 + 1_013_904_223, 2**31)
    assigned = bucket < 2**30
    rates = user_metrics[:, 1:].clone()
    rates[:, :16] /= user_metrics[:, :1].clamp_min(1.0)
    rates[:, 19:23] /= user_metrics[:, :1].clamp_min(1.0)
    rates[:, -1] /= user_metrics[:, 0].clamp_min(1.0)
    for cell, mask in enumerate((~assigned, assigned)):
        if cohort is not None:
            mask &= cohort
        values = rates[mask]
        cell_stats[cell, :, 0] += mask.sum()
        cell_stats[cell, :, 1] += values.sum(dim=0)
        cell_stats[cell, :, 2] += values.square().sum(dim=0)



def candidate_batch(config, generator, device, state, catalog, step, policy=None):
    del generator, device
    graph = build_candidate_graph(
        config, state, catalog, step,
        None if policy is None else getattr(policy, "enabled_routes", None),
    )
    item_ids = graph["item_ids"]
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
    if config.signal_version == "kuairand-calibrated-v3":
        features = build_tensor_features(
            config, state["user_ids"], state, batch, step
        )
        batch["stay_nonlinear"] = nonlinear_stay_adjustment(features)
    return batch


def sample_step(config, policy, generator, device, state, selected, step):
    del generator, device
    affinity = (selected["topics"] * state["interest"]).sum(dim=1)
    is_live = selected["content_type"] == 1 if policy.multi_queue else torch.zeros_like(affinity).bool()
    is_ad = selected["content_type"] == 2 if policy.multi_queue else torch.zeros_like(affinity).bool()
    response_affinity = affinity + 0.03 * is_live.float() - 0.10 * is_ad.float()
    feed = sample_response(
        state["user_ids"],
        step,
        config.seed,
        config.signal_version,
        state["active"],
        state["fatigue"],
        state["satisfaction"],
        response_affinity,
        selected["quality"],
        selected.get("duration"),
        selected.get("stay_nonlinear"),
    )
    stay, long_view, quality_view, like, negative, played, play_draw = feed
    local = sample_local_response(
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
    lt_value, feed_tree, local_tree, commercialization = business_and_lt_values(
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
        "detail": detail, "favorite": favorite, "paid": paid, "pixel": pixel,
        "ad_selected": is_ad,
        "effective_ad": effective_ad,
        "ad_contribution": ad_contribution,
        "organic_opportunity_cost": opportunity_cost,
        "live_selected": is_live,
        "selected_duration": selected["duration"],
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
        values["selected_duration"].sum(),
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
    totals = torch.zeros(29, dtype=torch.float64, device=device)
    cell_stats = torch.zeros(2, 26, 3, dtype=torch.float64, device=device)
    lt_exchange_stats = torch.zeros(2, 6, dtype=torch.float64, device=device)
    trigger_cell_stats = torch.zeros(2, 26, 3, dtype=torch.float64, device=device)
    trigger_users = torch.zeros((), dtype=torch.long, device=device)
    candidate_diagnostics = torch.zeros(
        5 + len(ROUTE_NAMES) + 2, dtype=torch.float64, device=device
    )
    trace_rows = []
    for offset in range(0, config.users, config.batch_users):
        users = min(config.batch_users, config.users - offset)
        user_ids = torch.arange(offset, offset + users, device=device, dtype=torch.int64)
        initial_policy = policy_schedule[0] if policy_schedule else policy
        state = new_user_state(
            config, initial_policy, generator, device, user_ids
        )
        user_metrics = torch.zeros(users, 27, device=device)
        trigger_cohort = torch.zeros(users, dtype=torch.bool, device=device)
        for step in range(config.steps):
            active_policy = (
                policy_schedule[step] if policy_schedule is not None else policy
            )
            refresh_search_state(config, state, step)
            if trigger_kind and step == measurement_start_step:
                trigger_cohort = trigger_mask(state, trigger_kind)
                trigger_users += trigger_cohort.sum()
            candidates = candidate_batch(
                config, generator, device, state, catalog, step, active_policy)
            selected = select_candidate(
                active_policy, user_ids, state, candidates, device, step, config
            )
            active_before = state["active"].clone()
            values = sample_step(
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
                    values["selected_duration"],
                ), dim=1).float()
            return_value, returned = advance_state(
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


def prepare_run(config, policy_schedule, measurement_start_step, trigger_kind):
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
    device, generator, catalog = prepare_run(
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
