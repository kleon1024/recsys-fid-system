"""Device-resident batched Feed trajectory and candidate-graph simulator."""

from __future__ import annotations

from time import perf_counter

import torch

from ...simulation.experimentation.assignment import assign_binary_torch
from ...value import BUSINESS_TREE_WEIGHTS, DEFAULT_LT_CONFIG
from .artifact.features import build_tensor_features
from .calibration.nonlinear import nonlinear_stay_adjustment
from .lt_exchange import (
    accumulate_lt_exchange_components,
)
from .tensor_catalog import build_tensor_catalog
from .graph.candidate import ROUTE_NAMES, build_candidate_graph
from .graph.reporting import CELL_METRICS, PER_EXPOSURE_METRICS, render_report
from .graph.trace import append_trace
from .graph.random import uniform_for_items
from .experiment.trigger import (
    combine_tensor_ab,
    combine_tensor_cuped_ab,
    combine_tensor_counterfactual_ab,
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
from .tensor_runtime.behavior.external import ExternalSequenceMixtureWorld
from .tensor_runtime.contracts import (
    DEFAULT_GPU_BATCH_USERS,
    EXTERNAL_MIXTURE_FEED_VERSION,
    LOCAL_NEURAL_SIGNAL_VERSION,
    TensorFeedConfig,
)
from .tensor_runtime.response import (
    business_and_lt_values,
    sample_response,
)
from .tensor_runtime.local_response import sample_local_response
from .tensor_runtime.state import (
    advance_state,
    new_user_state,
    sample_terminal_retention,
)


USER_METRIC_COLUMN = {
    name: index + 1 for index, name in enumerate(CELL_METRICS)
}


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
    "combine_tensor_cuped_ab",
    "combine_tensor_counterfactual_ab",
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



def _user_rates(user_metrics):
    rates = user_metrics[:, 1:].clone()
    exposure = user_metrics[:, :1].clamp_min(1.0)
    for index, name in enumerate(CELL_METRICS):
        if name in PER_EXPOSURE_METRICS:
            rates[:, index] /= exposure[:, 0]
    return rates


def _accumulate_cells(
    cell_stats, user_ids, user_metrics, cohort=None, assignment_version="legacy",
    experiment_salt=0x1B873593,
):
    if assignment_version == "avalanche_v2":
        assigned = assign_binary_torch(user_ids, experiment_salt)
    else:
        bucket = torch.remainder(
            user_ids * 1_664_525 + 1_013_904_223, 2**31
        )
        assigned = bucket < 2**30
    rates = _user_rates(user_metrics)
    for cell, mask in enumerate((~assigned, assigned)):
        if cohort is not None:
            mask &= cohort
        values = rates[mask]
        cell_stats[cell, :, 0] += mask.sum()
        cell_stats[cell, :, 1] += values.sum(dim=0)
        cell_stats[cell, :, 2] += values.square().sum(dim=0)


def _accumulate_batch_cells(
    config, cells, trigger_cells, user_ids, metrics, trigger_cohort,
    trigger_kind,
):
    version = (
        "avalanche_v2"
        if config.local_signal_version == LOCAL_NEURAL_SIGNAL_VERSION
        else "legacy"
    )
    _accumulate_cells(
        cells, user_ids, metrics, assignment_version=version,
        experiment_salt=config.experiment_salt,
    )
    if trigger_kind:
        _accumulate_cells(
            trigger_cells, user_ids, metrics, trigger_cohort,
            assignment_version=version,
            experiment_salt=config.experiment_salt,
        )



def candidate_batch(config, generator, device, state, catalog, step, policy=None):
    del generator, device
    graph = build_candidate_graph(
        config, state, catalog, step,
        None if policy is None else getattr(policy, "enabled_routes", None),
        None if policy is None else getattr(policy, "score_ann_pool", None),
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
        "creator_need": catalog.creator_need[item_ids],
        "duplicate_cluster": catalog.duplicate_cluster[item_ids],
        "latent_integrity_risk": catalog.latent_integrity_risk[item_ids],
        "predicted_integrity_risk": catalog.predicted_integrity_risk[item_ids],
        "latent_experience_quality": catalog.latent_experience_quality[item_ids],
    }
    if catalog.behavior_sparse is not None:
        batch["behavior_sparse"] = catalog.behavior_sparse[item_ids]
        batch["behavior_dense"] = catalog.behavior_dense[item_ids]
    batch.update(graph)
    if config.signal_version.startswith("kuairand-"):
        features = build_tensor_features(
            config, state["user_ids"], state, batch, step
        )
        batch["stay_nonlinear"] = nonlinear_stay_adjustment(features)
    return batch


def _served_candidate_values(selected, active):
    return {
        "selected_duration": selected["duration"] * active,
        "coarse_oracle_survives": (
            selected["coarse_oracle_survives"].float() * active
        ),
        "coarse_pass_fraction": selected["coarse_pass_fraction"] * active,
        "oracle_regret": selected["oracle_regret"] * active,
        "poi_candidate_fraction": selected["poi_candidate_fraction"] * active,
        "stage_attribution": selected["stage_attribution"],
        "route_valid_counts": selected["route_valid_counts"],
        "unique_recall_count": selected["unique_recall_count"],
        "predicted_integrity_risk": selected.get(
            "predicted_integrity_risk", torch.zeros_like(selected["quality"])
        ) * active,
        "near_duplicate": selected.get(
            "repeated_cluster", torch.zeros_like(active)
        ) & active,
        "repeated_author": selected.get(
            "repeated_author", torch.zeros_like(active)
        ) & active,
        "selected_poi": selected["is_poi"].bool() & active,
        "governance_eligible_fraction": selected.get(
            "governance_eligible_fraction", torch.ones_like(selected["quality"])
        ) * active,
    }


def sample_step(
    config, policy, generator, device, state, selected, step,
    behavior_world: ExternalSequenceMixtureWorld | None = None,
):
    del generator, device
    affinity = (selected["topics"] * state["interest"]).sum(dim=1)
    is_live = selected["content_type"] == 1 if policy.multi_queue else torch.zeros_like(affinity).bool()
    is_ad = selected["content_type"] == 2 if policy.multi_queue else torch.zeros_like(affinity).bool()
    response_affinity = affinity + 0.03 * is_live.float() - 0.10 * is_ad.float()
    if behavior_world is None:
        feed = sample_response(
            state["user_ids"],
            step,
            config.seed,
            config.signal_version,
            state["active"],
            state["fatigue"],
            state["satisfaction"],
            response_affinity,
            selected.get("latent_experience_quality", selected["quality"]),
            selected.get("duration"),
            selected.get("stay_nonlinear"),
        )
        stay, long_view, quality_view, like, negative, played, play_draw = feed
        feed_events = {
            "comment": torch.zeros_like(played),
            "share": torch.zeros_like(played),
            "follow": torch.zeros_like(played),
        }
    else:
        feed_events = behavior_world.sample(config, state, selected, step)
        stay = feed_events["stay"]
        long_view = feed_events["long_view"]
        quality_view = feed_events["quality_view"]
        like = feed_events["like"]
        negative = feed_events["negative"]
        played = feed_events["played"]
        play_draw = feed_events["play_draw"]
    local = sample_local_response(
        state["user_ids"],
        step,
        config.seed,
        config.local_signal_version,
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
    active = state["active"]
    selected_ad = is_ad & active
    selected_live = is_live & active
    effective_ad = is_ad & played
    ad_contribution = effective_ad.float() * selected["ad_value"]
    opportunity_cost = (
        selected_ad.float() * selected["organic_opportunity_cost"]
    )
    lt_value += (
        ad_contribution
        * DEFAULT_LT_CONFIG.rates["accepted_commercialization_unit"].unit_value
    )
    ads_live_tree = (
        ad_contribution * BUSINESS_TREE_WEIGHTS["ad_value"]
        + selected_live.float()
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
        "ad_selected": selected_ad,
        "effective_ad": effective_ad,
        "ad_contribution": ad_contribution,
        "organic_opportunity_cost": opportunity_cost,
        "live_selected": selected_live,
    }
    values.update(_served_candidate_values(selected, active))
    values.update({
        name: feed_events[name] for name in ("comment", "share", "follow")
    })
    if "history_item" in feed_events:
        values["history_item"] = feed_events["history_item"]
        values["history_feedback"] = feed_events["history_feedback"]
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
        values["predicted_integrity_risk"].sum(),
        values["near_duplicate"].sum(),
        values["repeated_author"].sum(), values["selected_poi"].sum(),
        values["governance_eligible_fraction"].sum(),
        values["comment"].sum(), values["share"].sum(),
        values["follow"].sum(),
        torch.zeros((), device=values["stay"].device),
    )).to(torch.float64)


def _maybe_append_trace(
    rows, config, state, candidates, selected, values, step, offset,
    measurement_start,
):
    inside_window = step < measurement_start + config.trace_requests_per_user
    if offset == 0 and inside_window:
        append_trace(rows, config, state, candidates, selected, values, step)


def _finalize_batch_metrics(
    config, cells, trigger_cells, lt_exchange, paired, all_users,
    preperiod_users, user_ids, user_metrics, preperiod_metrics,
    trigger_cohort, trigger_kind,
):
    _accumulate_batch_cells(
        config, cells, trigger_cells, user_ids, user_metrics,
        trigger_cohort, trigger_kind,
    )
    if config.retain_paired_user_metrics:
        rates = _user_rates(user_metrics)
        paired.append(
            rates[assign_binary_torch(user_ids, config.experiment_salt)].cpu()
        )
        all_users.append(rates.cpu())
        preperiod_users.append(_user_rates(preperiod_metrics).cpu())
    accumulate_lt_exchange_components(lt_exchange, user_ids, user_metrics)


def _measurement_user_values(active, values):
    return torch.stack((
        active, values["stay"], values["long_view"], values["quality_view"],
        values["negative"], values["lt_value"], values["local_value_tree"],
        values["anchor"], values["paid"] | values["pixel"],
        values["ad_selected"], values["effective_ad"],
        values["ad_contribution"], values["organic_opportunity_cost"],
        values["feed_value_tree"], values["ads_live_value_tree"],
        values["accepted_commercialization"],
        values["local_commercialization"], torch.zeros_like(values["stay"]),
        values["accepted_commercialization"], values["lt_value"],
        values["coarse_oracle_survives"], values["coarse_pass_fraction"],
        values["oracle_regret"], values["poi_candidate_fraction"],
        values["stay"] / 60.0
        * DEFAULT_LT_CONFIG.rates["stay_minute"].unit_value,
        torch.zeros_like(values["stay"]), values["selected_duration"],
        values["predicted_integrity_risk"], values["near_duplicate"],
        values["repeated_author"], values["selected_poi"],
        values["governance_eligible_fraction"],
        values["played"], (values["stay"] >= 3.0) & active.bool(),
        values["like"], values["comment"], values["share"], values["follow"],
    ), dim=1).float()


def _record_step(
    totals, diagnostics, trace_rows, config, state, candidates, selected,
    values, active, step, offset, measurement_start,
):
    totals += _step_totals(config, {**state, "active": active}, values)
    attribution = values["stage_attribution"][active]
    diagnostics[:5] += torch.bincount(
        attribution, minlength=5
    ).to(torch.float64)
    diagnostics[5 : 5 + len(ROUTE_NAMES)] += (
        values["route_valid_counts"][active].sum(dim=0).to(torch.float64)
    )
    diagnostics[-2] += values["unique_recall_count"][active].sum()
    diagnostics[-1] += active.sum()
    _maybe_append_trace(
        trace_rows, config, state, candidates, selected, values,
        step, offset, measurement_start,
    )
    return _measurement_user_values(active, values)


def _record_terminal_retention(totals, user_metrics, retained):
    active_day_value = (
        retained.float() * DEFAULT_LT_CONFIG.rates["active_day"].unit_value
    )
    totals[10] += active_day_value.sum()
    totals[37] += retained.sum()
    user_metrics[:, USER_METRIC_COLUMN[
        "lt_value_per_exposure"
    ]] += active_day_value
    user_metrics[:, USER_METRIC_COLUMN["active_days_per_user"]] += retained
    user_metrics[:, USER_METRIC_COLUMN["lt_value_per_user"]] += active_day_value
    user_metrics[:, USER_METRIC_COLUMN[
        "lt_active_days_per_user"
    ]] += active_day_value


def _record_return(totals, active, returned, first):
    totals[8] += active.sum() * int(first) + returned.sum()
    totals[9] += returned.sum()


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
    behavior_world,
):
    totals = torch.zeros(38, dtype=torch.float64, device=device)
    cell_stats = torch.zeros(
        2, len(CELL_METRICS), 3, dtype=torch.float64, device=device
    )
    lt_exchange_stats = torch.zeros(2, 6, dtype=torch.float64, device=device)
    trigger_cell_stats = torch.zeros(
        2, len(CELL_METRICS), 3, dtype=torch.float64, device=device
    )
    trigger_users = torch.zeros((), dtype=torch.long, device=device)
    candidate_diagnostics = torch.zeros(
        5 + len(ROUTE_NAMES) + 2, dtype=torch.float64, device=device
    )
    trace_rows = []
    paired_user_metrics = []
    all_user_metrics = []
    preperiod_user_metrics = []
    for offset in range(0, config.users, config.batch_users):
        users = min(config.batch_users, config.users - offset)
        user_ids = torch.arange(offset, offset + users, device=device, dtype=torch.int64)
        initial_policy = policy_schedule[0] if policy_schedule else policy
        state = new_user_state(
            config, initial_policy, generator, device, user_ids
        )
        if behavior_world is not None:
            behavior_world.initialize_state(state)
        metric_columns = len(CELL_METRICS) + 1
        user_metrics = torch.zeros(users, metric_columns, device=device)
        preperiod_metrics = torch.zeros(users, metric_columns, device=device)
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
                config, active_policy, generator, device, state, selected, step,
                behavior_world,
            )
            measured = step >= measurement_start_step
            if measured:
                user_metrics += _record_step(
                    totals, candidate_diagnostics, trace_rows, config, state,
                    candidates, selected, values, active_before, step, offset,
                    measurement_start_step,
                )
            else:
                preperiod_metrics += _measurement_user_values(
                    active_before, values
                )
            _, returned = advance_state(
                config, active_policy, generator, state, selected, values, step
            )
            if measured:
                _record_return(
                    totals, active_before, returned,
                    step == measurement_start_step,
                )
        _record_terminal_retention(
            totals, user_metrics, sample_terminal_retention(config, state)
        )
        _finalize_batch_metrics(
            config, cell_stats, trigger_cell_stats, lt_exchange_stats,
            paired_user_metrics, all_user_metrics, preperiod_user_metrics,
            user_ids, user_metrics, preperiod_metrics, trigger_cohort,
            trigger_kind,
        )
    return (
        totals,
        cell_stats,
        lt_exchange_stats,
        trigger_cell_stats,
        trigger_users,
        candidate_diagnostics,
        trace_rows,
        paired_user_metrics,
        all_user_metrics,
        preperiod_user_metrics,
    )


def prepare_run(
    config, policy_schedule, measurement_start_step, trigger_kind,
    behavior_world: ExternalSequenceMixtureWorld | None = None,
):
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
    catalog_seed = config.seed if config.catalog_seed is None else config.catalog_seed
    catalog_generator = torch.Generator(device=device).manual_seed(catalog_seed)
    catalog = build_tensor_catalog(config, catalog_generator, device)
    if behavior_world is not None:
        catalog = behavior_world.attach_catalog(catalog, config)
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
    behavior_world: ExternalSequenceMixtureWorld | None = None,
) -> dict[str, object]:
    if (
        config.signal_version == EXTERNAL_MIXTURE_FEED_VERSION
        and behavior_world is None
    ):
        raise ValueError("external Feed V4 requires an evidence-bound behavior world")
    device, generator, catalog = prepare_run(
        config, policy_schedule, measurement_start_step, trigger_kind,
        behavior_world,
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
        behavior_world,
    )
    (
        totals,
        cell_stats,
        lt_exchange_stats,
        trigger_stats,
        trigger_users,
        candidate_diagnostics,
        trace_rows,
        paired_user_metrics,
        all_user_metrics,
        preperiod_user_metrics,
    ) = simulation
    _sync(device)
    seconds = perf_counter() - started
    report = render_report(
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
    if paired_user_metrics:
        report["_paired_user_metrics"] = torch.cat(paired_user_metrics)
        report["_all_user_metrics"] = torch.cat(all_user_metrics)
        report["_preperiod_user_metrics"] = torch.cat(preperiod_user_metrics)
    if behavior_world is not None:
        report["behavior_world"] = behavior_world.describe()
    return report
