"""Hidden and observed user-state transitions after one Feed exposure."""

from __future__ import annotations

import torch

from ....value import DEFAULT_LT_CONFIG
from ..graph.random import uniform
from .ranking_sequence import append_ranking_event


def _update_response_state(state, selected, values):
    engagement = values["long_view"].float() + values["like"].float()
    engagement += 0.5 * values.get("comment", torch.zeros_like(engagement)).float()
    engagement += 0.7 * values.get("share", torch.zeros_like(engagement)).float()
    engagement += 0.8 * values.get("follow", torch.zeros_like(engagement)).float()
    satisfaction = torch.clamp(
        0.84 * state["hidden_satisfaction"] + 0.09 * engagement
        - 0.28 * values["negative"].float(), -1.0, 1.0,
    )
    repetition = state["topic_counts"].gather(
        1, selected["candidate_topic"][:, None]
    ).squeeze(1) / state["topic_counts"].sum(dim=1).clamp_min(1.0)
    fatigue = torch.clamp(
        0.76 * state["hidden_fatigue"]
        + 0.05 * values["long_view"].float() + 0.12 * repetition
        - 0.05 * values["negative"].float(), 0.0, 1.0,
    )
    state["hidden_satisfaction"], state["hidden_fatigue"] = satisfaction, fatigue
    state["satisfaction"] = torch.clamp(
        0.88 * state["satisfaction"] + 0.12 * satisfaction, -1.0, 1.0
    )
    state["fatigue"] = torch.clamp(
        0.84 * state["fatigue"] + 0.16 * fatigue, 0.0, 1.0
    )
    return satisfaction, fatigue


def _update_interests(policy, state, selected, values):
    update = values["long_view"].float()[:, None]
    state["interest"] = torch.nn.functional.normalize(
        state["interest"] * (1.0 - 0.10 * update)
        + selected["topics"] * 0.10 * update, dim=1,
    )
    if "hidden_drift_target" in state:
        drift = 0.0025 + 0.0040 * state["hidden_novelty"][:, None]
        state["interest"] = torch.nn.functional.normalize(
            state["interest"] * (1.0 - drift)
            + state["hidden_drift_target"] * drift, dim=1,
        )
    for name in ("observed_interest", "local_observed_interest"):
        state[name] = torch.nn.functional.normalize(
            state[name] * (1.0 - policy.realtime_interest_rate * update)
            + selected["topics"] * policy.realtime_interest_rate * update, dim=1,
        )


def _update_history(state, values):
    if "history_item" not in values:
        return
    for name in ("behavior_history_items", "behavior_history_feedback"):
        state[name] = torch.roll(state[name], shifts=-1, dims=1)
    state["behavior_history_items"][:, -1] = values["history_item"]
    state["behavior_history_feedback"][:, -1] = values[
        "history_feedback"
    ].to(torch.uint8)


def _advance_session(config, state, step, satisfaction, fatigue):
    state["requests_in_session"] += state["active"]
    leave = (
        uniform(state["user_ids"], step, 50, config.seed)
        < torch.sigmoid(-3.4 - 1.2 * satisfaction + 1.7 * fatigue)
    ) & state["active"]
    if config.signal_version == "heterogeneous-nonlinear-v2":
        leave |= (
            state["requests_in_session"] >= config.requests_per_session
        ) & state["active"]
    returned = leave & (
        uniform(state["user_ids"], step, 51, config.seed)
        < torch.sigmoid(1.0 + 1.6 * satisfaction - 1.1 * fatigue)
    ) & (state["sessions"] < config.max_sessions)
    state["returns"] += returned
    state["sessions"] += returned
    state["requests_in_session"] = torch.where(
        returned, torch.zeros_like(state["requests_in_session"]),
        state["requests_in_session"],
    )
    for name in ("ads_served", "live_served"):
        state[name] = torch.where(returned, torch.zeros_like(state[name]), state[name])
    state["last_ad_step"] = torch.where(
        returned, torch.full_like(state["last_ad_step"], -10_000),
        state["last_ad_step"],
    )
    state["active"] &= ~leave | returned
    return returned


def advance_state(config, policy, generator, state, selected, values, step):
    del generator
    satisfaction, fatigue = _update_response_state(state, selected, values)
    _update_interests(policy, state, selected, values)
    state["retarget_item"] = torch.where(
        values["anchor"], selected["item_ids"], state["retarget_item"]
    )
    active = state["active"].float()
    state["topic_counts"].scatter_add_(
        1, selected["candidate_topic"][:, None], active[:, None]
    )
    state["last_topic"] = torch.where(
        state["active"], selected["candidate_topic"], state["last_topic"]
    )
    state["ads_served"] += values["ad_selected"] & state["active"]
    state["live_served"] += values["live_selected"] & state["active"]
    state["last_ad_step"] = torch.where(
        values["ad_selected"] & state["active"],
        torch.full_like(state["last_ad_step"], step), state["last_ad_step"],
    )
    _update_history(state, values)
    append_ranking_event(state, selected, values)
    state["search_strength"] *= 0.78
    if config.search_event_rate > 0.0:
        state["search_ttl"] = torch.clamp(state["search_ttl"] - 1, min=0)
        state["search_strength"] *= (state["search_ttl"] > 0).float()
    returned = _advance_session(config, state, step, satisfaction, fatigue)
    return (
        returned.float() * DEFAULT_LT_CONFIG.rates["active_day"].unit_value,
        returned,
    )
