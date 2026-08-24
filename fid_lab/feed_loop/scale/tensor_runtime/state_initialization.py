"""Vectorized persistent user and observable-state initialization."""

from __future__ import annotations

import torch

from ..graph.random import normal, uniform
from .contracts import EXTERNAL_MIXTURE_FEED_VERSION


def _interests(config, policy, user_ids):
    if config.signal_version == "heterogeneous-nonlinear-v2":
        interest = (-torch.log(uniform(
            user_ids, 0, 1, config.seed, config.topics
        ).clamp_min(1e-7))).pow(1.0 / 0.8)
        interest = torch.nn.functional.normalize(interest, dim=1)
        observed = torch.clamp(
            interest + policy.observation_noise
            * normal(user_ids, 0, 3, config.seed, config.topics), min=0.0,
        )
    else:
        interest = torch.nn.functional.normalize(
            normal(user_ids, 0, 5, config.seed, config.topics), dim=1
        )
        observed = interest + policy.observation_noise * normal(
            user_ids, 0, 7, config.seed, config.topics
        )
    return interest, torch.nn.functional.normalize(observed, dim=1)


def _profile(config, user_ids, interest):
    age = torch.floor(uniform(user_ids, 0, 16, config.seed).square() * 3_650.0)
    activity = torch.floor(
        (0.15 + 0.85 * uniform(user_ids, 0, 17, config.seed))
        * torch.sqrt(age + 1.0) * 3.2
    ).clamp_max(200.0)
    lifecycle = torch.where(
        age < 7, torch.zeros_like(age),
        torch.where(
            age < 30, torch.ones_like(age),
            torch.where(age < 365, torch.full_like(age, 2), torch.full_like(age, 3)),
        ),
    ).long()
    return {
        "account_age_days": age,
        "historical_activity": activity,
        "lifecycle_bucket": lifecycle,
        "region_bucket": torch.floor(
            uniform(user_ids, 0, 18, config.seed) * 10
        ).long(),
        "topic_counts": torch.softmax(interest.abs() * 2.0, dim=1)
        * activity[:, None],
    }


def _external_state(config, user_ids, users, device):
    return {
        "hidden_mixture": torch.floor(
            uniform(user_ids, 0, 105, config.seed) * 4
        ).long(),
        "hidden_novelty": uniform(user_ids, 0, 107, config.seed),
        "hidden_patience": uniform(user_ids, 0, 109, config.seed),
        "hidden_drift_target": torch.nn.functional.normalize(
            normal(user_ids, 0, 111, config.seed, config.topics), dim=1
        ),
        "behavior_history_items": torch.zeros(
            users, config.behavior_sequence_length, device=device, dtype=torch.long
        ),
        "behavior_history_feedback": torch.zeros(
            users, config.behavior_sequence_length, 7,
            device=device, dtype=torch.uint8,
        ),
    }


def new_user_state(config, policy, generator, device, user_ids):
    del generator
    users = len(user_ids)
    interest, observed = _interests(config, policy, user_ids)
    external = config.signal_version == EXTERNAL_MIXTURE_FEED_VERSION
    state = {
        "user_ids": user_ids,
        "eligible": (
            torch.remainder(user_ids * 1_103_515_245 + 12_345, 2**31).float()
            / float(2**31) < policy.eligible_fraction
        ).float()[:, None],
        "interest": interest,
        "observed_interest": observed,
        "local_observed_interest": torch.nn.functional.normalize(
            interest + policy.local_observation_noise
            * normal(user_ids, 0, 11, config.seed, config.topics), dim=1,
        ),
        "satisfaction": torch.zeros(users, device=device),
        "fatigue": torch.zeros(users, device=device),
        "hidden_satisfaction": (
            0.12 * normal(user_ids, 0, 101, config.seed)
            if external else torch.zeros(users, device=device)
        ),
        "hidden_fatigue": (
            0.10 * uniform(user_ids, 0, 103, config.seed)
            if external else torch.zeros(users, device=device)
        ),
        "active": torch.ones(users, dtype=torch.bool, device=device),
        "sessions": torch.ones(users, device=device),
        "requests_in_session": torch.zeros(users, device=device),
        "returns": torch.zeros(users, device=device),
        "search_topic": torch.floor(
            uniform(user_ids, 0, 13, config.seed) * config.topics
        ).long(),
        "search_strength": (
            torch.zeros(users, device=device) if config.search_event_rate > 0.0
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
        "ads_served": torch.zeros(users, device=device, dtype=torch.long),
        "live_served": torch.zeros(users, device=device, dtype=torch.long),
        "last_ad_step": torch.full((users,), -10_000, device=device, dtype=torch.long),
        **_profile(config, user_ids, interest),
    }
    if external:
        state.update(_external_state(config, user_ids, users, device))
    return state
