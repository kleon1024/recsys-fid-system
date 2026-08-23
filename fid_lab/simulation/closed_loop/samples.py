"""Point-in-time training rows, event closure, and Joiner reports."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ...evolution.data.contracts import (
    ActionEvent, CommerceEvent, OutboundClick, PixelEvent, StageDecision,
    synthetic_impression_time,
)
from ...evolution.data.joiner import EvolutionJoiner
from ...evolution.data.request_dataset import build_request_candidate_dataset
from ..contracts import SimulationConfig, Trajectory
from ..policies import HeuristicPolicy
from ..population import run_population


def training_data(config: SimulationConfig, catalog):
    policy = HeuristicPolicy()
    trajectories = run_population(
        config, catalog, policy, range(config.users), explore=True
    )
    rows = [row for trajectory in trajectories for row in trajectory.rows]
    return rows, *example_arrays(rows, config)


def example_arrays(rows, config: SimulationConfig):
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    labels = np.asarray([row.response.long_view for row in rows], dtype=np.int8)
    users = np.asarray([row.user_id for row in rows], dtype=np.int64)
    sessions = np.asarray(
        [row.user_id * config.max_sessions + row.session_id for row in rows], dtype=np.int64
    )
    propensities = np.asarray([row.selection_probability for row in rows], dtype=np.float32)
    return features, labels, users, sessions, propensities


def row_decisions(
    row,
    catalog,
    model_name: str,
    model_manifest,
    history,
    timestamp: int,
    observable: bool,
):
    order = np.argsort(-np.asarray(row.candidate_scores))
    rank_by_index = np.empty(len(order), dtype=int)
    rank_by_index[order] = np.arange(1, len(order) + 1)
    decisions = []
    for candidate_index, (candidate_id, features, score, propensity, routes) in enumerate(
        zip(
            row.candidate_ids,
            row.candidate_features,
            row.candidate_scores,
            row.candidate_propensities,
            row.candidate_routes,
        )
    ):
        decisions.append(
            StageDecision(
                row.request_id,
                row.user_id,
                int(catalog.author[candidate_id]),
                candidate_id,
                int(catalog.poi[candidate_id]),
                timestamp,
                int(catalog.category[candidate_id]),
                int(catalog.city[candidate_id]),
                (
                    row.user_id % 1024,
                    int(catalog.author[candidate_id]),
                    candidate_id,
                    int(catalog.poi[candidate_id]),
                    int(catalog.category[candidate_id]),
                    int(catalog.city[candidate_id]),
                ),
                features,
                tuple(history[-8:]),
                "+".join(routes),
                propensity,
                score,
                int(rank_by_index[candidate_index]),
                candidate_id == row.item_id,
                observable,
                {"fine": score, "value_tree": score},
                {"model": model_name, **model_manifest},
            )
        )
    return decisions


def row_events(row, catalog, timestamp: int, observable: bool):
    actions = []
    commerce = []
    clicks = []
    pixels = []
    item_id = row.item_id
    response = row.response
    action_specs = (
        ("play", response.play, 1, 1.0),
        ("play_3s", response.play_3s, 3, 1.0),
        ("stay_seconds", response.play, 4, response.stay_seconds),
        ("slide", response.slide, 5, 1.0),
        ("long_view", response.long_view, 30, 1.0),
        ("high_quality_long_view", response.high_quality_long_view, 31, 1.0),
        ("like", response.like, 35, 1.0),
        ("favorite", response.favorite, 36, 1.0),
        ("comment", response.comment, 37, 1.0),
        ("share", response.share, 38, 1.0),
        ("negative_feedback", response.negative_feedback, 39, 1.0),
        ("anchor_impression", response.anchor_impression, 40, 1.0),
        ("anchor_click", response.anchor_click, 41, 1.0),
        ("poi_detail", response.poi_detail, 50, 1.0),
        ("poi_favorite", response.poi_favorite, 70, 1.0),
    )
    actions.extend(
        ActionEvent(
            f"{row.request_id}-{name}", row.request_id, item_id,
            int(catalog.poi[item_id]), name, timestamp + delay, timestamp + delay + 2,
            value,
        )
        for name, happened, delay, value in action_specs
        if happened
    )
    open_loop = int(catalog.fulfillment[item_id]) == 2
    if response.order and not open_loop:
        commerce_specs = (("order", response.order, 300), ("payment", response.payment, 360))
        commerce.extend(
            CommerceEvent(
                f"{row.request_id}-{name}", row.request_id, item_id,
                int(catalog.poi[item_id]), name, f"o-{row.request_id}",
                f"p-{row.request_id}" if response.payment else None,
                timestamp + delay, timestamp + delay + 3
            )
            for name, happened, delay in commerce_specs
            if happened
        )
    if response.order and open_loop:
        click_id = f"click-{row.request_id}"
        identity = f"identity-{row.user_id}" if observable else None
        clicks.append(
            OutboundClick(
                click_id, row.request_id, item_id, int(catalog.poi[item_id]), identity,
                int(catalog.author[item_id]), timestamp + 100
            )
        )
        if response.pixel_conversion:
            pixels.append(
                PixelEvent(
                    f"pixel-{row.request_id}", f"conversion-{row.request_id}", identity,
                    int(catalog.author[item_id]), timestamp + 900, timestamp + 930,
                    click_id if observable else None
                )
            )
    return actions, commerce, clicks, pixels


def build_feed_joiner(
    config: SimulationConfig,
    catalog,
    observed: list[Trajectory],
    policies,
    assigned: np.ndarray,
):
    decisions: list[StageDecision] = []
    actions: list[ActionEvent] = []
    commerce: list[CommerceEvent] = []
    clicks: list[OutboundClick] = []
    pixels: list[PixelEvent] = []
    for user_index, trajectory in enumerate(observed[: config.joiner_users]):
        policy = policies[int(assigned[user_index])]
        model_manifest = {
            "feature": (
                "stateful-v2"
                if config.signal_version == "heterogeneous-nonlinear-v2"
                else "stateful-v1"
            ),
            "code": "simulation-v2",
            "signal_version": config.signal_version,
            **dict(getattr(policy, "artifact_manifest", {})),
        }
        history: list[tuple[float, ...]] = []
        for row in trajectory.rows:
            timestamp = synthetic_impression_time(
                user_index, row.session_id, row.request_index
            )
            observable = user_index % 10 != 0
            decisions.extend(
                row_decisions(
                    row,
                    catalog,
                    policy.name,
                    model_manifest,
                    history,
                    timestamp,
                    observable,
                )
            )
            row_actions, row_commerce, row_clicks, row_pixels = row_events(
                row, catalog, timestamp, observable
            )
            actions.extend(row_actions)
            commerce.extend(row_commerce)
            clicks.extend(row_clicks)
            pixels.extend(row_pixels)
            history.append(row.features[:8])
    report = EvolutionJoiner().build(
        decisions, actions, commerce, clicks, pixels, watermark=1_900_000_000
    )
    return report


def joiner_report(config, catalog, observed, policies, assigned):
    report = build_feed_joiner(config, catalog, observed, policies, assigned)
    request_dataset = build_request_candidate_dataset(
        observed[: config.joiner_users],
        catalog,
        report,
        {"schema": "request-candidate-v1", "source": "stateful-feed"},
    )
    tasks = tuple(report.fine[0].labels) if report.fine else ()
    label_rates = {}
    mask_coverage = {}
    for task in tasks:
        eligible = [example for example in report.fine if example.label_masks[task]]
        mask_coverage[task] = len(eligible) / len(report.fine)
        label_rates[task] = (
            float(np.mean([example.labels[task] for example in eligible]))
            if eligible
            else None
        )
    return {
        "stage_decisions": len(report.coarse),
        "recall_examples": len(report.recall),
        "coarse_examples": len(report.coarse),
        "fine_examples": len(report.fine),
        "duplicate_events": report.duplicate_events,
        "orphan_events": report.orphan_events,
        "immature_task_labels": report.immature_task_labels,
        "fine_label_rates": label_rates,
        "fine_label_mask_coverage": mask_coverage,
        "pixel_attribution": asdict(report.attribution),
        "request_candidate_dataset": {
            "requests": len(request_dataset.requests),
            "candidate_decisions": len(request_dataset.candidates),
            "mature_label_rows": len(request_dataset.labels),
            "one_exposure_per_request": (
                sum(
                    value.exposed_position == 1
                    for value in request_dataset.candidates
                )
                == len(request_dataset.requests)
            ),
            "stage_attribution": request_dataset.stage_attribution,
        },
    }


