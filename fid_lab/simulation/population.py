"""Batched policy serving over independent stateful user trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..value import (
    BusinessValueSignals,
    BusinessValueTree,
    LTMetricContainer,
    LTMetricVector,
)
from .contracts import TraceRow, Trajectory
from .environment import StatefulFeedEnv
from .policies import ParameterizedPolicy


@dataclass
class _Accumulator:
    rows: list[TraceRow] = field(default_factory=list)
    sessions: int = 1
    returned_sessions: int = 0
    plays: int = 0
    play_3s: int = 0
    stay_seconds: float = 0.0
    slides: int = 0
    long_views: int = 0
    high_quality_long_views: int = 0
    likes: int = 0
    favorites: int = 0
    comments: int = 0
    shares: int = 0
    watch_minutes: float = 0.0
    anchor_impressions: int = 0
    anchor_clicks: int = 0
    poi_details: int = 0
    poi_favorites: int = 0
    orders: int = 0
    negative_feedback: int = 0
    lt_value: float = 0.0
    local_value_tree_score: float = 0.0
    lt_components: dict[str, float] = field(
        default_factory=lambda: {
            "stay": 0.0,
            "active_days": 0.0,
            "accepted_commercialization": 0.0,
        }
    )
    business_value_components: dict[str, float] = field(
        default_factory=lambda: {
            "feed": 0.0,
            "local_consumption": 0.0,
            "local_transaction": 0.0,
            "local_supply": 0.0,
            "ads_live": 0.0,
        }
    )

    def finish(self) -> Trajectory:
        return Trajectory(
            tuple(self.rows),
            self.sessions,
            self.returned_sessions,
            self.plays,
            self.play_3s,
            self.stay_seconds,
            self.slides,
            self.long_views,
            self.high_quality_long_views,
            self.likes,
            self.favorites,
            self.comments,
            self.shares,
            self.watch_minutes,
            self.anchor_impressions,
            self.anchor_clicks,
            self.poi_details,
            self.poi_favorites,
            self.orders,
            self.negative_feedback,
            self.lt_value,
            self.local_value_tree_score,
            dict(self.lt_components),
            dict(self.business_value_components),
        )


def _selection(
    scores: np.ndarray,
    exploration_rate: float,
    seed: int,
) -> tuple[int, np.ndarray]:
    greedy = int(np.argmax(scores))
    propensities = np.zeros(len(scores), dtype=float)
    if exploration_rate <= 0.0:
        propensities[greedy] = 1.0
        return greedy, propensities
    propensities.fill(exploration_rate / len(scores))
    propensities[greedy] += 1.0 - exploration_rate
    action = int(np.random.default_rng(seed).choice(len(scores), p=propensities))
    return action, propensities


def _record_values(environment, item_id, response, returned, accumulator) -> None:
    is_open_loop = int(environment.catalog.fulfillment[item_id]) == 2
    margin = float(environment.catalog.commerce_value[item_id]) * 8.0
    local_commercialization = (
        margin
        if (response.payment and not is_open_loop)
        or (response.pixel_conversion and is_open_loop)
        else 0.0
    )
    business_value = BusinessValueTree().evaluate(
        BusinessValueSignals(
            quality_view=int(response.high_quality_long_view),
            meaningful_engagements=sum(
                map(
                    int,
                    (
                        response.high_quality_long_view,
                        response.like,
                        response.favorite,
                        response.comment,
                        response.share,
                    ),
                )
            ),
            negative_feedback=int(response.negative_feedback),
            anchor_click=int(response.anchor_click),
            poi_detail=int(response.poi_detail),
            poi_favorite=int(response.poi_favorite),
            closed_loop_payment=int(response.payment and not is_open_loop),
            open_loop_verified_conversion=float(
                response.pixel_conversion and is_open_loop
            ),
            contribution_margin=local_commercialization,
        )
    )
    lt_value = LTMetricContainer().evaluate(
        LTMetricVector(
            stay_minutes=response.watch_minutes,
            active_days=float(returned),
            accepted_commercialization_value=0.0,
        )
    )
    accumulator.lt_value += lt_value.total
    accumulator.local_value_tree_score += (
        business_value.local_consumption + business_value.local_transaction
    )
    for name in accumulator.lt_components:
        accumulator.lt_components[name] += float(getattr(lt_value, name))
    for name in accumulator.business_value_components:
        accumulator.business_value_components[name] += float(
            getattr(business_value, name)
        )


def _record_step(
    environment: StatefulFeedEnv,
    observation: np.ndarray,
    scores: np.ndarray,
    exploration_rate: float,
    accumulator: _Accumulator,
):
    session_id = environment.session_id
    request_index = environment.request_index
    candidate_ids = environment.candidates.copy()
    query_embedding = tuple(float(value) for value in environment.observed_interest)
    action, propensities = _selection(
        scores,
        exploration_rate,
        environment.config.seed
        + environment.user_id * 65_537
        + session_id * 101
        + request_index,
    )
    oracle = tuple(
        float(environment._behavior_probabilities(features, int(item_id))["long_view"])
        for features, item_id in zip(observation, candidate_ids)
    )
    next_observation, _, terminated, _, info = environment.step(action)
    response = info["response"]
    item_id = int(info["item_id"])
    returned = bool(info["session_end"] and info["returned"])
    request_id = f"u{environment.user_id}-s{session_id}-r{request_index}"
    accumulator.rows.append(
        TraceRow(
            environment.user_id,
            session_id,
            request_index,
            request_id,
            int(candidate_ids[action]),
            tuple(int(value) for value in candidate_ids),
            tuple(tuple(float(value) for value in candidate) for candidate in observation),
            tuple(float(value) for value in scores),
            tuple(float(value) for value in propensities),
            oracle,
            environment.candidate_routes,
            environment.recall_count,
            environment.coarse_count,
            tuple(float(value) for value in observation[action]),
            float(scores[action]),
            float(propensities[action]),
            response,
            returned,
            environment.parameter_snapshot,
            query_embedding,
        )
    )
    accumulator.watch_minutes += response.watch_minutes
    accumulator.plays += int(response.play)
    accumulator.play_3s += int(response.play_3s)
    accumulator.stay_seconds += response.stay_seconds
    accumulator.slides += int(response.slide)
    accumulator.long_views += int(response.long_view)
    accumulator.high_quality_long_views += int(response.high_quality_long_view)
    accumulator.likes += int(response.like)
    accumulator.favorites += int(response.favorite)
    accumulator.comments += int(response.comment)
    accumulator.shares += int(response.share)
    accumulator.anchor_impressions += int(response.anchor_impression)
    accumulator.anchor_clicks += int(response.anchor_click)
    accumulator.poi_details += int(response.poi_detail)
    accumulator.poi_favorites += int(response.poi_favorite)
    accumulator.orders += int(response.payment)
    accumulator.negative_feedback += int(response.negative_feedback)
    _record_values(environment, item_id, response, returned, accumulator)
    if returned:
        accumulator.returned_sessions += 1
        accumulator.sessions += 1
    return next_observation, terminated


def run_population(
    config,
    catalog,
    policy,
    user_ids,
    explore: bool = False,
    parameters=None,
    retrieval_snapshot=None,
):
    """Batch model inference by request depth while preserving per-user dynamics."""
    users = [int(user_id) for user_id in user_ids]
    serving_policy = ParameterizedPolicy(policy, parameters) if parameters else policy
    environments = [
        StatefulFeedEnv(config, catalog, parameters, retrieval_snapshot) for _ in users
    ]
    observations = []
    for environment, user_id in zip(environments, users):
        observation, _ = environment.reset(
            seed=config.seed + user_id, options={"user_id": user_id}
        )
        observations.append(observation)
    accumulators = [_Accumulator() for _ in users]
    active = np.ones(len(users), dtype=bool)
    exploration_rate = config.exploration_rate if explore else 0.0
    while active.any():
        indices = np.flatnonzero(active)
        batch = np.concatenate([observations[index] for index in indices], axis=0)
        batch_scores = serving_policy.score(batch).reshape(
            len(indices), config.candidates
        )
        for row_index, user_index in enumerate(indices):
            observation, terminated = _record_step(
                environments[user_index],
                observations[user_index],
                batch_scores[row_index],
                exploration_rate,
                accumulators[user_index],
            )
            observations[user_index] = observation
            active[user_index] = not terminated
    return [accumulator.finish() for accumulator in accumulators]
