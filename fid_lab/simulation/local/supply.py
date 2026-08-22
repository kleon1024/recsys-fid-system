"""Posting-side POI selection and supply-to-distribution feedback."""

from __future__ import annotations

from dataclasses import replace
from math import exp

import numpy as np

from ..ab import experiment_metrics, launch_decision, randomization_audit
from ..contracts import Catalog, PostingResponse, Trajectory
from ..policies import LocalConstrainedValuePolicy
from ..population import run_population


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-max(min(value, 20.0), -20.0)))


def simulate_posting(
    trajectories: list[Trajectory],
    catalog: Catalog,
    personalized: bool,
    seed: int,
) -> list[PostingResponse]:
    responses = []
    for trajectory in trajectories:
        user_id = trajectory.rows[0].user_id
        rng = np.random.default_rng(seed + user_id * 17_389)
        local_intent = (
            trajectory.anchor_clicks
            + 1.5 * trajectory.poi_details
            + 2.0 * trajectory.poi_favorites
        )
        entered = bool(rng.random() < _sigmoid(-3.6 + 0.28 * local_intent))
        candidate_items = [
            row.item_id for row in trajectory.rows if catalog.is_poi_video[row.item_id]
        ]
        if not candidate_items:
            candidate_items = [trajectory.rows[-1].item_id]
        selected_item = max(
            candidate_items,
            key=lambda item: (
                catalog.quality[item]
                + (0.35 * catalog.freshness[item] if personalized else 0.0)
            ),
        )
        selection_logit = -1.2 + 0.9 * float(catalog.quality[selected_item])
        if personalized:
            selection_logit += 0.55 + 0.15 * local_intent
        selected = bool(entered and rng.random() < _sigmoid(selection_logit))
        submitted = bool(
            selected and rng.random() < (0.78 if personalized else 0.60)
        )
        published = bool(
            submitted and rng.random() < (0.95 if personalized else 0.90)
        )
        quality = float(
            np.clip(
                0.45
                + (0.08 if personalized else 0.0)
                + 0.05 * trajectory.high_quality_long_views
                + 0.04 * trajectory.likes
                + rng.normal(0.0, 0.08),
                0.05,
                0.98,
            )
        )
        responses.append(
            PostingResponse(
                entered,
                10 if entered else 0,
                selected,
                submitted,
                published,
                int(catalog.poi[selected_item]) if selected else None,
                quality,
            )
        )
    return responses


def apply_published_supply(
    catalog: Catalog,
    responses: list[PostingResponse],
) -> Catalog:
    published = [response for response in responses if response.published]
    count = min(len(published), int((~catalog.is_poi_video).sum()))
    if count == 0:
        return catalog
    available = np.flatnonzero(~catalog.is_poi_video)
    selected = available[np.argsort(catalog.freshness[available])[:count]]
    is_poi = catalog.is_poi_video.copy()
    freshness = catalog.freshness.copy()
    quality = catalog.quality.copy()
    is_poi[selected] = True
    freshness[selected] = 1.0
    quality[selected] = np.asarray(
        [response.predicted_content_quality for response in published[:count]],
        dtype=np.float32,
    )
    return replace(
        catalog,
        is_poi_video=is_poi,
        freshness=freshness,
        quality=quality,
    )


def posting_summary(responses: list[PostingResponse]) -> dict[str, float | int]:
    users = len(responses)
    return {
        "users": users,
        "entry_rate": sum(value.entered_posting_page for value in responses) / users,
        "poi_selection_rate": sum(value.poi_selected for value in responses) / users,
        "submit_rate": sum(value.submitted for value in responses) / users,
        "publish_rate": sum(value.published for value in responses) / users,
        "published_videos": sum(value.published for value in responses),
        "mean_published_quality": float(
            np.mean(
                [
                    value.predicted_content_quality
                    for value in responses
                    if value.published
                ]
                or [0.0]
            )
        ),
    }


def _paired_world_effects(control_distribution, treatment_distribution):
    paired_world_effects = {}
    for metric, attribute in {
        "stay_seconds": "stay_seconds",
        "lt_views": "long_views",
        "hlt_views": "high_quality_long_views",
        "anchor_clicks": "anchor_clicks",
        "local_service_value": "local_service_value",
        "long_term_value": "discounted_value",
    }.items():
        control = np.asarray(
            [getattr(value, attribute) for value in control_distribution], dtype=float
        )
        treatment = np.asarray(
            [getattr(value, attribute) for value in treatment_distribution], dtype=float
        )
        absolute = float((treatment - control).mean())
        control_mean = float(control.mean())
        paired_world_effects[metric] = {
            "control_mean": control_mean,
            "treatment_mean": float(treatment.mean()),
            "absolute_effect": absolute,
            "relative_effect": None
            if abs(control_mean) < 1e-9
            else absolute / abs(control_mean),
        }
    return paired_world_effects


def run_supply_iteration(config, catalog, source_trajectories, distribution_policy):
    control_posting = simulate_posting(
        source_trajectories, catalog, personalized=False, seed=config.seed + 501
    )
    treatment_posting = simulate_posting(
        source_trajectories, catalog, personalized=True, seed=config.seed + 501
    )
    control_catalog = apply_published_supply(catalog, control_posting)
    treatment_catalog = apply_published_supply(catalog, treatment_posting)
    users = np.arange(config.users) + 40_000_000
    control_distribution = run_population(config, control_catalog, distribution_policy, users)
    treatment_distribution = run_population(
        config, treatment_catalog, distribution_policy, users
    )
    value_policy = LocalConstrainedValuePolicy(distribution_policy)
    value_distribution = run_population(config, treatment_catalog, value_policy, users)
    assigned = np.random.default_rng(config.seed + 503).random(config.users) < 0.5
    value_metrics, value_potential = experiment_metrics(
        treatment_distribution, value_distribution, assigned
    )
    return {
        "experiment_unit": "city-day switchback or author cluster",
        "user_randomization_valid": False,
        "interference_reason": "Published supply changes other users' candidate corpus.",
        "control_posting": posting_summary(control_posting),
        "treatment_posting": posting_summary(treatment_posting),
        "base_catalog_poi_items": int(catalog.is_poi_video.sum()),
        "control_catalog_poi_items": int(control_catalog.is_poi_video.sum()),
        "treatment_catalog_poi_items": int(treatment_catalog.is_poi_video.sum()),
        "supply_only_paired_world_effects": _paired_world_effects(
            control_distribution, treatment_distribution
        ),
        "value_tree_on_fixed_catalog_ab": {
            "experiment_unit": "viewer_id",
            "metrics": value_metrics,
            "randomization_audit": randomization_audit(
                value_potential, config.seed + 1503
            ),
            "decision": launch_decision(value_metrics),
            "config": {
                "local_weight": value_policy.local_weight,
                "maximum_feed_score_loss": value_policy.maximum_feed_score_loss,
            },
        },
        "value_tree_increment_paired_effects": _paired_world_effects(
            treatment_distribution, value_distribution
        ),
        "combined_supply_and_value_tree_effects": _paired_world_effects(
            control_distribution, value_distribution
        ),
    }
