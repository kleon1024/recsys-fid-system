"""Capacity and common-random A/B review for the external Feed V4 world."""

from __future__ import annotations

from dataclasses import asdict, replace

import torch

from ....tensor_cascade import select_candidate
from ....tensor_policies import PERSONALIZED, POPULAR
from ...tensor_engine import (
    candidate_batch,
    combine_tensor_counterfactual_ab,
    prepare_run,
    run_tensor_feed,
)
from ..state import new_user_state


def _context_capacity(config, world, users=20_000):
    sample = replace(
        config, users=min(config.users, users),
        batch_users=min(config.users, users), retain_paired_user_metrics=False,
    )
    device, generator, catalog = prepare_run(sample, None, 0, None, world)
    user_ids = torch.arange(sample.users, device=device)
    state = new_user_state(sample, PERSONALIZED, generator, device, user_ids)
    world.initialize_state(state)
    candidates = candidate_batch(
        sample, generator, device, state, catalog, 0, PERSONALIZED
    )
    selected = select_candidate(
        PERSONALIZED, user_ids, state, candidates, device, 0, sample
    )
    base, base_stay, _ = world.predict(state, selected, 0)
    shuffled = {
        **state,
        "behavior_history_items": torch.roll(
            state["behavior_history_items"], shifts=1, dims=0
        ),
        "behavior_history_feedback": torch.roll(
            state["behavior_history_feedback"], shifts=1, dims=0
        ),
    }
    changed, changed_stay, _ = world.predict(shuffled, selected, 0)
    task_movement = (base - changed).abs().mean(dim=0)
    mixture = {}
    for cohort in range(4):
        mask = state["hidden_mixture"] == cohort
        mixture[str(cohort)] = {
            "users": int(mask.sum()),
            "play": float(base[mask, 0].mean()),
            "long_view": float(base[mask, 1].mean()),
            "like": float(base[mask, 2].mean()),
            "negative": float(base[mask, 6].mean()),
        }
    return {
        "requests": sample.users,
        "history_permutation_mean_absolute_probability_delta": float(
            task_movement.mean()
        ),
        "history_permutation_long_view_delta": float(task_movement[1]),
        "history_permutation_stay_delta": float(
            (base_stay - changed_stay).abs().mean()
        ),
        "mixture_response_means": mixture,
    }


def _range(values, key):
    selected = [row[key] for row in values.values()]
    return max(selected) - min(selected)


def run_feed_behavior_review(config, world):
    capacity = _context_capacity(config, world)
    control = run_tensor_feed(config, POPULAR, behavior_world=world)
    treatment = run_tensor_feed(config, PERSONALIZED, behavior_world=world)
    ab = combine_tensor_counterfactual_ab(control, treatment)
    rates = treatment["metrics"]
    mixture = capacity["mixture_response_means"]
    gates = {
        "artifact_lineage_bound": (
            control["behavior_world"] == treatment["behavior_world"]
            == world.describe()
        ),
        "sequence_context_material": (
            capacity["history_permutation_long_view_delta"] >= 0.005
        ),
        "mixture_response_heterogeneous": _range(mixture, "long_view") >= 0.002,
        "behavior_rates_supported": (
            0.05 <= rates["play_rate"] <= 0.30
            and 0.005 <= rates["long_view_rate"] <= 0.15
            and rates["negative_rate"] <= 0.01
        ),
        "personalization_stay_positive": (
            ab["stay_per_exposure"]["confidence_interval"][0] > 0
        ),
        "personalization_lt_positive": (
            ab["lt_value_per_user"]["confidence_interval"][0] > 0
        ),
        "quality_view_nonnegative": (
            ab["quality_long_view_rate"]["confidence_interval"][0] >= 0
        ),
        "negative_guardrail": (
            ab["negative_rate"]["confidence_interval"][1] <= 0.0002
        ),
    }
    for report in (control, treatment):
        report.pop("_paired_user_metrics", None)
    return {
        "schema": "external-mixture-feed-v4-launch-review-v1",
        "config": asdict(config),
        "behavior_world": world.describe(),
        "capacity": capacity,
        "control": control,
        "treatment": treatment,
        "paired_ab": ab,
        "gates": gates,
        "decision": "feed_v4_behavior_pass" if all(gates.values())
        else "hold_feed_v4_behavior",
        "evidence_boundary": (
            "External randomized-sequence calibration plus synthetic hidden "
            "population mixtures. This validates simulator behavior and policy "
            "effect recovery; it is not production lift."
        ),
    }
