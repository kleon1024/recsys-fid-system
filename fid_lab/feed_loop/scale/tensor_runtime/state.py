"""Vectorized user state initialization and transition ownership."""

from __future__ import annotations

import torch

from ....value import DEFAULT_LT_CONFIG
from ..graph.random import normal, uniform


def new_user_state(config, policy, generator, device, user_ids):
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
    age_draw = uniform(user_ids, 0, 16, config.seed)
    account_age_days = torch.floor(age_draw.square() * 3_650.0)
    historical_activity = torch.floor(
        (0.15 + 0.85 * uniform(user_ids, 0, 17, config.seed))
        * torch.sqrt(account_age_days + 1.0)
        * 3.2
    ).clamp_max(200.0)
    lifecycle_bucket = torch.where(
        account_age_days < 7,
        torch.zeros_like(account_age_days),
        torch.where(
            account_age_days < 30,
            torch.ones_like(account_age_days),
            torch.where(
                account_age_days < 365,
                torch.full_like(account_age_days, 2),
                torch.full_like(account_age_days, 3),
            ),
        ),
    ).long()
    region_bucket = torch.floor(uniform(user_ids, 0, 18, config.seed) * 10).long()
    history_profile = torch.softmax(interest.abs() * 2.0, dim=1)
    historical_topic_counts = history_profile * historical_activity[:, None]
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
        "topic_counts": historical_topic_counts,
        "account_age_days": account_age_days,
        "historical_activity": historical_activity,
        "lifecycle_bucket": lifecycle_bucket,
        "region_bucket": region_bucket,
        "ads_served": torch.zeros(users, device=device, dtype=torch.long),
        "live_served": torch.zeros(users, device=device, dtype=torch.long),
        "last_ad_step": torch.full((users,), -10_000, device=device, dtype=torch.long),
    }


def advance_state(config, policy, generator, state, selected, values, step):
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

