"""Stable report rendering for tensor Feed simulations."""

from __future__ import annotations

from dataclasses import asdict

import torch

from ....value import DEFAULT_LT_CONFIG
from ..lt_exchange import render_lt_exchange_components
from .candidate import ROUTE_NAMES
from .trace import render_trace


CELL_METRICS = (
    "stay_per_exposure", "long_view_rate", "quality_long_view_rate",
    "negative_rate", "lt_value_per_exposure",
    "local_value_tree_score_per_exposure", "anchor_click_rate",
    "conversion_rate", "ad_load", "effective_ad_load",
    "ad_contribution_per_exposure", "organic_opportunity_cost_per_exposure",
    "feed_value_tree_score_per_exposure", "ads_live_value_tree_score_per_exposure",
    "accepted_platform_commercialization_per_exposure",
    "local_commercialization_value_per_exposure", "active_days_per_user",
    "accepted_platform_commercialization_per_user", "lt_value_per_user",
    "coarse_feed_oracle_recall", "coarse_pass_fraction",
    "fine_oracle_regret_per_exposure", "poi_candidate_fraction",
    "lt_stay_per_user", "lt_active_days_per_user",
)
STAGE_NAMES = (
    "recall_miss", "coarse_miss", "fine_rank_miss", "mix_rank_miss",
    "served_audit_oracle",
)


def render_cells(cell_stats):
    report = {}
    for cell, cell_name in enumerate(("control", "treatment")):
        report[cell_name] = {}
        for metric, name in enumerate(CELL_METRICS):
            count, total, total_square = cell_stats[cell, metric]
            mean = total / count
            variance = torch.clamp(
                (total_square - total.square() / count) / (count - 1.0),
                min=0.0,
            )
            report[cell_name][name] = {
                "users": int(count),
                "mean": float(mean),
                "variance": float(variance),
            }
    return report


def render_trigger(kind, users, stats, population):
    if not kind:
        return None
    return {
        "kind": kind,
        "eligible_users": int(users),
        "eligible_rate": float(users / population),
        "cells": render_cells(stats.cpu()),
    }


def render_candidate_graph(candidate_diagnostics):
    values = candidate_diagnostics.cpu().numpy()
    requests = max(values[-1], 1.0)
    return {
        "stage_attribution": {
            name: int(values[index]) for index, name in enumerate(STAGE_NAMES)
        },
        "stage_attribution_rate": {
            name: float(values[index] / requests)
            for index, name in enumerate(STAGE_NAMES)
        },
        "mean_route_hits": {
            name: float(values[5 + index] / requests)
            for index, name in enumerate(ROUTE_NAMES)
        },
        "mean_unique_recalled": float(values[-2] / requests),
        "requests": int(values[-1]),
    }


def render_metrics(values, config):
    exposures = max(values[0], 1.0)
    lt_rates = DEFAULT_LT_CONFIG.rates
    return {
        "exposures": int(values[0]),
        "stay_per_exposure": float(values[1] / exposures),
        "long_view_rate": float(values[2] / exposures),
        "quality_long_view_rate": float(values[3] / exposures),
        "like_rate": float(values[4] / exposures),
        "negative_rate": float(values[5] / exposures),
        "play_rate": float(values[6] / exposures),
        "play_3s_rate": float(values[7] / exposures),
        "sessions_per_user": float(values[8] / config.users),
        "returned_sessions_per_user": float(values[9] / config.users),
        "active_days_per_user": float(values[9] / config.users),
        "lt_value_per_exposure": float(values[10] / exposures),
        "lt_value_per_user": float(values[10] / config.users),
        "lt_stay_per_user": float(
            values[1] / 60.0 * lt_rates["stay_minute"].unit_value / config.users
        ),
        "lt_active_days_per_user": float(
            values[9] * lt_rates["active_day"].unit_value / config.users
        ),
        "local_value_tree_score_per_exposure": float(values[11] / exposures),
        "anchor_click_rate": float(values[12] / exposures),
        "poi_detail_rate": float(values[13] / exposures),
        "closed_loop_payment_rate": float(values[14] / exposures),
        "open_loop_conversion_rate": float(values[15] / exposures),
        "conversion_rate": float((values[14] + values[15]) / exposures),
        "ad_load": float(values[16] / exposures),
        "effective_ad_load": float(values[17] / exposures),
        "ad_contribution_per_exposure": float(values[18] / exposures),
        "organic_opportunity_cost_per_exposure": float(values[19] / exposures),
        "feed_value_tree_score_per_exposure": float(values[20] / exposures),
        "ads_live_value_tree_score_per_exposure": float(values[21] / exposures),
        "accepted_platform_commercialization_per_exposure": float(
            values[22] / exposures
        ),
        "accepted_platform_commercialization_per_user": float(
            values[22] / config.users
        ),
        "local_commercialization_value_per_exposure": float(values[23] / exposures),
        "coarse_feed_oracle_recall": float(values[24] / exposures),
        "coarse_pass_fraction": float(values[25] / exposures),
        "fine_oracle_regret_per_exposure": float(values[26] / exposures),
        "poi_candidate_fraction": float(values[27] / exposures),
    }


def render_report(
    config, policy, schedule, measurement_start, totals, cells, lt_exchange,
    trigger_kind, trigger_users, trigger_stats, candidate_diagnostics, trace_rows,
    seconds, device,
):
    values = totals.cpu().numpy()
    report = {
        "config": asdict(config),
        "policy": policy.describe() if hasattr(policy, "describe") else asdict(policy),
        "policy_schedule": None if schedule is None else [value.name for value in schedule],
        "measurement_start_step": measurement_start,
        "experiment_cells": render_cells(cells.cpu()),
        "lt_exchange_components": render_lt_exchange_components(lt_exchange.cpu()),
        "candidate_graph": {
            "version": config.candidate_graph_version,
            **render_candidate_graph(candidate_diagnostics),
        },
        "request_candidate_trace": render_trace(trace_rows),
        "metrics": render_metrics(values, config),
        "performance": {
            "seconds": seconds,
            "users_per_second": config.users / seconds,
            "requests_per_second": values[0] / seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device.index or 0))
                if device.type == "cuda" else 0
            ),
        },
    }
    trigger = render_trigger(trigger_kind, trigger_users, trigger_stats, config.users)
    if trigger:
        report["trigger_experiment"] = trigger
    return report
