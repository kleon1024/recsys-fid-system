"""Cross-day coupled rollout of consumer response and creator supply."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt

import torch

from ...simulation.randomness import uniform
from ...value import DEFAULT_LT_CONFIG
from ..scale.experiment.trigger import refresh_search_state
from ..scale.tensor_engine import candidate_batch, prepare_run, sample_step
from ..scale.tensor_runtime.state import advance_state, new_user_state
from ..tensor_cascade import select_candidate
from .contracts import CONSUMER_GUARDRAILS, CREATOR_RETENTION_GUARDRAILS
from .creators import (
    CreatorFeedback,
    CreatorResponseWorld,
    initialize_creators,
    refresh_catalog,
)


USER_METRICS = (
    "exposures", "stay", "long_view", "negative", "lt", "returned_days",
)


@dataclass
class UserPartition:
    state: dict[str, torch.Tensor]
    totals: torch.Tensor


def _move(state, device):
    return {name: value.to(device) for name, value in state.items()}


def _initialize_partitions(feed_config, policy, generator, device, world):
    partitions = []
    for offset in range(0, feed_config.users, feed_config.batch_users):
        users = min(feed_config.batch_users, feed_config.users - offset)
        ids = torch.arange(offset, offset + users, device=device)
        state = new_user_state(feed_config, policy, generator, device, ids)
        world.initialize_state(state)
        partitions.append(UserPartition(
            _move(state, "cpu"), torch.zeros(users, len(USER_METRICS))
        ))
    return partitions


def _reactivate(state, day, seed):
    inactive = ~state["active"]
    probability = torch.sigmoid(
        -1.2 + 1.4 * state["hidden_satisfaction"]
        - 1.0 * state["hidden_fatigue"] + 0.5 * state["hidden_patience"]
    )
    returned = inactive & (
        uniform(state["user_ids"], day, 451, seed) < probability
    )
    state["active"] |= returned
    state["sessions"] += returned
    state["requests_in_session"] = torch.where(
        returned, torch.zeros_like(state["requests_in_session"]),
        state["requests_in_session"],
    )
    return returned


def _accumulate_user(totals, active, values, return_value, returned):
    totals[:, 0] += active
    totals[:, 1] += values["stay"]
    totals[:, 2] += values["long_view"]
    totals[:, 3] += values["negative"]
    totals[:, 4] += values["lt_value"] + return_value
    totals[:, 5] += returned


def _simulate_partition(
    partition, feed_config, ecosystem, policy, generator, device, catalog,
    behavior_world, creator_feedback, day,
):
    state = _move(partition.state, device)
    totals = partition.totals.to(device)
    day_returned = _reactivate(state, day, ecosystem.seed) if day else torch.zeros(
        len(totals), dtype=torch.bool, device=device
    )
    if day:
        totals[:, 4] += day_returned * DEFAULT_LT_CONFIG.rates[
            "active_day"
        ].unit_value
        totals[:, 5] += day_returned
    day_values = torch.zeros(6, dtype=torch.float64, device=device)
    for request in range(ecosystem.steps_per_day):
        step = day * ecosystem.steps_per_day + request
        refresh_search_state(feed_config, state, step)
        candidates = candidate_batch(
            feed_config, generator, device, state, catalog, step, policy
        )
        selected = select_candidate(
            policy, state["user_ids"], state, candidates, device, step,
            feed_config,
        )
        active = state["active"].clone()
        values = sample_step(
            feed_config, policy, generator, device, state, selected, step,
            behavior_world,
        )
        creator_feedback.add(selected["author"], values, active)
        return_value, returned = advance_state(
            feed_config, policy, generator, state, selected, values, step
        )
        _accumulate_user(totals, active, values, return_value, returned)
        day_values += torch.stack((
            active.sum(), values["stay"].sum(), values["long_view"].sum(),
            values["negative"].sum(), values["lt_value"].sum(),
            returned.sum(),
        )).double()
    partition.state = _move(state, "cpu")
    partition.totals = totals.cpu()
    return day_values


def _daily_report(day, values, feedback, population, publishers):
    exposures = values[0].clamp_min(1.0)
    active_creators = population.active.sum().clamp_min(1)
    creator_exposed = (feedback.exposures > 0).sum()
    return {
        "day": day,
        "exposures": int(values[0]),
        "stay_per_exposure": float(values[1] / exposures),
        "long_view_rate": float(values[2] / exposures),
        "negative_rate": float(values[3] / exposures),
        "lt_per_exposure": float(values[4] / exposures),
        "returned_sessions": int(values[5]),
        "active_creators": int(active_creators),
        "creator_retention_rate": float(active_creators / len(population.active)),
        "creators_with_distribution": int(creator_exposed),
        "creator_distribution_coverage": float(
            creator_exposed / len(population.active)
        ),
        "new_supply": int(len(publishers)),
        "mean_new_supply_quality": (
            float(population.quality[publishers].mean()) if len(publishers) else 0.0
        ),
    }


def _run_arm(feed_config, ecosystem, policy, behavior_world):
    device, generator, catalog = prepare_run(
        feed_config, None, 0, None, behavior_world
    )
    partitions = _initialize_partitions(
        feed_config, policy, generator, device, behavior_world
    )
    population = initialize_creators(
        catalog, feed_config.catalog_creators, ecosystem.seed
    )
    creator_world = CreatorResponseWorld(device, ecosystem.seed)
    days = []
    for day in range(ecosystem.days):
        feedback = CreatorFeedback.empty(feed_config.catalog_creators, device)
        values = torch.zeros(6, dtype=torch.float64, device=device)
        for partition in partitions:
            values += _simulate_partition(
                partition, feed_config, ecosystem, policy, generator, device,
                catalog, behavior_world, feedback, day,
            )
        publishers = creator_world.advance(
            population, feedback, day, ecosystem.seed,
            ecosystem.max_new_items_per_day,
        )
        catalog = refresh_catalog(
            catalog, population, publishers, day, ecosystem.seed, behavior_world
        )
        days.append(_daily_report(day, values, feedback, population, publishers))
    user_totals = torch.cat(tuple(partition.totals for partition in partitions))
    creator_totals = torch.stack((
        population.active.float(), population.cumulative_posts,
        population.cumulative_retained_days, population.quality,
    ), dim=1).cpu()
    return {
        "policy": policy.name,
        "days": days,
        "users": feed_config.users,
        "creators": feed_config.catalog_creators,
        "behavior_world": behavior_world.describe(),
    }, user_totals, creator_totals


def _paired(control, treatment, names):
    delta = treatment.double() - control.double()
    output = {}
    for index, name in enumerate(names):
        values = delta[:, index]
        effect = float(values.mean())
        standard_error = float(values.std(unbiased=True) / sqrt(len(values)))
        interval = (effect - 1.96 * standard_error, effect + 1.96 * standard_error)
        output[name] = {
            "control_mean": float(control[:, index].mean()),
            "treatment_mean": float(treatment[:, index].mean()),
            "absolute_delta": effect,
            "confidence_interval": interval,
            "p_value": erfc(abs(effect / max(standard_error, 1e-12)) / sqrt(2.0)),
            "estimator": "paired_potential_outcome",
        }
    return output


def run_ecosystem(feed_config, ecosystem, control_policy, treatment_policy, world):
    control, control_users, control_creators = _run_arm(
        feed_config, ecosystem, control_policy, world
    )
    treatment, treatment_users, treatment_creators = _run_arm(
        feed_config, ecosystem, treatment_policy, world
    )
    user_ab = _paired(control_users, treatment_users, USER_METRICS)
    creator_ab = _paired(
        control_creators, treatment_creators,
        ("active", "cumulative_posts", "retained_days", "quality"),
    )
    if ecosystem.objective == "consumer":
        threshold = CONSUMER_GUARDRAILS
        gates = {
            "stay_primary_positive": user_ab["stay"]["confidence_interval"][0] > 0,
            "lt_primary_positive": user_ab["lt"]["confidence_interval"][0] > 0,
            "creator_supply_noninferior": creator_ab["cumulative_posts"][
                "confidence_interval"
            ][0] >= threshold["creator_posts_absolute"],
            "creator_retention_noninferior": creator_ab["active"][
                "confidence_interval"
            ][0] >= threshold["creator_active_absolute"],
        }
    else:
        threshold = CREATOR_RETENTION_GUARDRAILS
        gates = {
            "creator_retention_primary_positive": creator_ab["active"][
                "confidence_interval"
            ][0] > 0.0,
            "creator_posts_noninferior": creator_ab["cumulative_posts"][
                "confidence_interval"
            ][0] >= threshold["creator_posts_absolute"],
            "stay_noninferior": user_ab["stay"]["confidence_interval"][0]
            >= threshold["stay_seconds_per_user"],
            "lt_noninferior": user_ab["lt"]["confidence_interval"][0]
            >= threshold["lt_per_user"],
        }
    gates.update({
        "negative_guardrail": user_ab["negative"]["confidence_interval"][1]
        <= threshold["negative_events_per_user"],
        "new_supply_observed": sum(
            day["new_supply"] for day in treatment["days"]
        ) > 0,
    })
    return {
        "schema": "feed-creator-ecosystem-v4-launch-review-v1",
        "feed_config": asdict(feed_config),
        "ecosystem_config": asdict(ecosystem),
        "control": control,
        "treatment": treatment,
        "user_paired_ab": user_ab,
        "creator_paired_ab": creator_ab,
        "gates": gates,
        "decision": "ecosystem_v4_pass" if all(gates.values())
        else "hold_ecosystem_v4",
        "evidence_boundary": (
            "External consumer behavior plus synthetic neural creator response. "
            "This is ecosystem simulator evidence, not production lift."
        ),
    }
