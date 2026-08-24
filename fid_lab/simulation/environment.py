"""Gymnasium environment with request, session, and cross-session state."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from math import erfc, exp, log, sqrt

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .cascade import CascadeCandidateProvider
from .contracts import Catalog, Response, SimulationConfig
from .experimentation.contracts import FeedParameters


FEATURE_NAMES = (
    "estimated_interest_affinity",
    "item_quality",
    "commerce_value",
    "item_popularity",
    "short_sequence_match",
    "same_city",
    "realtime_satisfaction_proxy",
    "realtime_fatigue_proxy",
    "trust",
    "commerce_propensity",
    "session_progress",
    "long_sequence_match",
    "duration_log_norm",
    "poi_video_indicator",
    "user_bucket_norm",
    "item_bucket_norm",
    "author_bucket_norm",
    "category_norm",
    "post_search_match",
    "retarget_match",
    "poi_quality",
    "inventory_available",
    "distance_score",
    "local_interest_affinity",
    "account_age_norm",
    "historical_activity_norm",
    "lifecycle_bucket_norm",
    "region_bucket_norm",
)


def build_catalog(config: SimulationConfig) -> Catalog:
    rng = np.random.default_rng(config.seed)
    topics = rng.gamma(0.7, 1.0, size=(config.items, config.topics)).astype(np.float32)
    topics /= np.maximum(np.linalg.norm(topics, axis=1, keepdims=True), 1e-8)
    is_poi_video = rng.random(config.items) < 0.28
    fulfillment = np.zeros(config.items, dtype=np.int8)
    fulfillment[is_poi_video] = rng.choice((1, 2), size=int(is_poi_video.sum()), p=(0.65, 0.35))
    return Catalog(
        topics=topics,
        quality=rng.beta(3.0, 2.2, config.items).astype(np.float32),
        commerce_value=rng.beta(1.4, 5.0, config.items).astype(np.float32),
        popularity=rng.beta(2.0, 6.0, config.items).astype(np.float32),
        freshness=rng.beta(1.6, 3.0, config.items).astype(np.float32),
        duration_seconds=np.clip(
            rng.lognormal(3.2, 0.65, config.items), 5.0, 180.0
        ).astype(np.float32),
        is_poi_video=is_poi_video,
        category=np.argmax(topics, axis=1).astype(np.int32),
        city=rng.integers(0, 100, config.items, dtype=np.int32),
        author=rng.integers(0, max(config.items // 8, 1), config.items, dtype=np.int32),
        poi=rng.integers(0, max(config.items // 4, 1), config.items, dtype=np.int32),
        poi_quality=rng.beta(3.2, 2.0, config.items).astype(np.float32),
        inventory_available=(rng.random(config.items) < 0.92),
        fulfillment=fulfillment,
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-max(min(value, 20.0), -20.0)))


class StatefulFeedEnv(gym.Env):
    """POI Feed dynamics on the standard Gymnasium environment contract."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SimulationConfig,
        catalog: Catalog,
        parameters: FeedParameters | None = None,
        retrieval_snapshot=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.catalog = catalog
        self.parameters = parameters
        self.parameter_snapshot = asdict(parameters) if parameters is not None else None
        self.candidate_provider = CascadeCandidateProvider(
            config, catalog, parameters, retrieval_snapshot
        )
        self.action_space = spaces.Discrete(config.candidates)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.candidates, len(FEATURE_NAMES)),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.user_id = int(options.get("user_id", 0))
        user_rng = np.random.default_rng(self.config.seed + self.user_id * 104_729)
        self.base_interest = user_rng.gamma(0.8, 1.0, self.config.topics).astype(np.float32)
        self.base_interest /= np.linalg.norm(self.base_interest)
        self.interest = self.base_interest.copy()
        observed_noise = user_rng.normal(0.0, 0.45, self.config.topics).astype(np.float32)
        self.observed_interest = np.maximum(self.base_interest + observed_noise, 0.0)
        self.observed_interest /= np.linalg.norm(self.observed_interest)
        local_noise = user_rng.normal(0.0, 0.15, self.config.topics).astype(np.float32)
        self.local_observed_interest = np.maximum(
            self.base_interest + local_noise, 0.0
        )
        self.local_observed_interest /= np.linalg.norm(
            self.local_observed_interest
        )
        self.satisfaction = float(user_rng.uniform(-0.05, 0.05))
        self.fatigue = 0.0
        self.observed_satisfaction = 0.0
        self.observed_fatigue = 0.0
        self.trust = float(user_rng.beta(5.0, 2.0))
        self.commerce_propensity = float(user_rng.beta(1.5, 5.0))
        self.observed_trust = float(np.clip(self.trust + user_rng.normal(0.0, 0.12), 0.0, 1.0))
        self.observed_commerce_propensity = float(
            np.clip(self.commerce_propensity + user_rng.normal(0.0, 0.10), 0.0, 1.0)
        )
        self.city = int(user_rng.integers(0, 100))
        search_distribution = self.base_interest / self.base_interest.sum()
        self.search_topic = int(user_rng.choice(self.config.topics, p=search_distribution))
        self.search_strength = float(user_rng.beta(2.0, 3.0))
        self.recent_poi_ids: deque[int] = deque(maxlen=8)
        self.session_id = 0
        self.request_index = 0
        self.recent_topics: deque[int] = deque(maxlen=8)
        self.recent_item_ids: deque[int] = deque(maxlen=32)
        self._refresh_candidates()
        return self._observation(), {"session_id": 0, "request_index": 0}

    def _random(self, stream: int) -> np.random.Generator:
        seed = (
            self.config.seed
            + self.user_id * 104_729
            + self.session_id * 10_007
            + self.request_index * 503
            + stream * 7_919
        )
        return np.random.default_rng(seed)

    def _refresh_candidates(self) -> None:
        batch = self.candidate_provider.recall(self)
        self.candidates = batch.item_ids
        self.candidate_routes = batch.routes
        self.recall_count = batch.recall_count
        self.coarse_count = batch.coarse_count
        self.recalled_candidate_ids = batch.recalled_item_ids
        self.recalled_candidate_routes = batch.recalled_routes
        self.recall_merge_scores = batch.recall_scores
        self.recalled_coarse_scores = batch.coarse_scores
        self.recalled_synthetic_oracle_scores = batch.synthetic_oracle_scores
        self.corpus_oracle_item_id = batch.corpus_oracle_item_id

    def _observation(self) -> np.ndarray:
        item_topics = self.catalog.topics[self.candidates]
        affinity = item_topics @ self.observed_interest
        feature_noise = self._random(5).normal(0.0, 0.06, len(self.candidates))
        affinity = np.clip(affinity + feature_noise, -1.0, 1.0)
        local_affinity = item_topics @ self.local_observed_interest
        local_affinity = np.clip(
            local_affinity
            + self._random(6).normal(0.0, 0.03, len(self.candidates)),
            -1.0,
            1.0,
        )
        categories = self.catalog.category[self.candidates]
        recent = np.fromiter(self.recent_topics, dtype=np.int32)
        short_match = (
            np.ones(len(categories), dtype=np.float32)
            if not len(recent)
            else np.isin(categories, recent[-3:]).astype(np.float32)
        )
        long_match = (
            np.zeros(len(categories), dtype=np.float32)
            if not len(recent)
            else np.asarray([np.mean(recent == category) for category in categories], dtype=np.float32)
        )
        search_match = self.catalog.category[self.candidates] == self.search_topic
        candidate_pois = self.catalog.poi[self.candidates]
        retarget_match = np.isin(candidate_pois, np.asarray(self.recent_poi_ids, dtype=np.int32))
        distance_score = np.where(
            self.catalog.city[self.candidates] == self.city,
            1.0,
            0.05,
        )
        age_draw = ((self.user_id * 48_271 + 17) % 10_007) / 10_007.0
        account_age_days = age_draw * age_draw * 3_650.0
        historical_activity = min(
            200.0,
            (0.15 + 0.85 * self.observed_trust)
            * np.sqrt(account_age_days + 1.0)
            * 3.2,
        )
        lifecycle = (
            0 if account_age_days < 7 else
            1 if account_age_days < 30 else
            2 if account_age_days < 365 else 3
        )
        return np.column_stack(
            (
                affinity,
                self.catalog.quality[self.candidates],
                self.catalog.commerce_value[self.candidates],
                self.catalog.popularity[self.candidates],
                short_match,
                (self.catalog.city[self.candidates] == self.city).astype(np.float32),
                np.full(self.config.candidates, self.observed_satisfaction, dtype=np.float32),
                np.full(self.config.candidates, self.observed_fatigue, dtype=np.float32),
                np.full(self.config.candidates, self.observed_trust, dtype=np.float32),
                np.full(
                    self.config.candidates, self.observed_commerce_propensity, dtype=np.float32
                ),
                np.full(
                    self.config.candidates,
                    self.request_index / self.config.requests_per_session,
                    dtype=np.float32,
                ),
                long_match,
                np.log1p(self.catalog.duration_seconds[self.candidates])
                / np.log(181.0),
                self.catalog.is_poi_video[self.candidates].astype(np.float32),
                np.full(
                    self.config.candidates,
                    (self.user_id % 1024) / 1023.0,
                    dtype=np.float32,
                ),
                (self.candidates % 4096).astype(np.float32) / 4095.0,
                (self.catalog.author[self.candidates] % 1024).astype(np.float32)
                / 1023.0,
                self.catalog.category[self.candidates].astype(np.float32)
                / max(self.config.topics - 1, 1),
                search_match.astype(np.float32) * self.search_strength,
                retarget_match.astype(np.float32),
                self.catalog.poi_quality[self.candidates],
                self.catalog.inventory_available[self.candidates].astype(np.float32),
                distance_score.astype(np.float32),
                local_affinity,
                np.full(self.config.candidates, account_age_days / 3_650.0),
                np.full(self.config.candidates, historical_activity / 200.0),
                np.full(self.config.candidates, lifecycle / 3.0),
                np.full(self.config.candidates, (self.city // 10) / 9.0),
            )
        ).astype(np.float32)

    def _behavior_probabilities(self, features: np.ndarray, item_id: int) -> dict[str, float]:
        quality = float(features[1])
        value = float(features[2])
        short_match = float(features[4])
        same_city = float(features[5])
        long_match = float(features[11])
        affinity = float(self.catalog.topics[item_id] @ self.interest)
        novelty = 1.0 - long_match
        satisfaction = self.satisfaction
        fatigue = self.fatigue
        trust = self.trust
        commerce = self.commerce_propensity
        post_search_match = float(features[18])
        retarget_match = float(features[19])
        poi_quality = float(features[20])
        inventory = float(features[21])
        distance_score = float(features[22])
        stay_nonlinear, local_nonlinear, negative_nonlinear = (
            self._nonlinear_adjustments(features)
        )
        stay_log_mean = (
            0.45
            + 1.7 * affinity
            + 0.55 * quality
            + 0.25 * short_match
            + 0.20 * novelty
            + 0.20 * satisfaction
            - fatigue
            + stay_nonlinear
        )
        p_anchor = _sigmoid(
            -5.0 + 1.7 * affinity + 0.7 * same_city + 0.7 * trust + 0.5 * value
            + 1.4 * post_search_match + 1.1 * retarget_match + 0.5 * distance_score
            + local_nonlinear
        )
        p_detail = _sigmoid(
            -1.3 + 1.0 * affinity + 0.7 * quality + 0.6 * same_city
            + 0.8 * poi_quality + 0.7 * post_search_match
        )
        p_favorite = _sigmoid(-5.0 + 1.8 * affinity + 0.8 * quality)
        p_order = _sigmoid(
            -4.5 + 1.2 * value + 1.0 * commerce + 0.6 * trust
            + 0.9 * poi_quality + 0.9 * retarget_match + 1.2 * inventory
            + 0.35 * local_nonlinear
        )
        p_negative = _sigmoid(
            -5.0 - 1.7 * affinity - 0.8 * quality + 2.0 * fatigue
            - satisfaction + negative_nonlinear
        )
        p_play = _sigmoid(3.0 + 0.4 * affinity - 0.6 * fatigue)
        duration = float(self.catalog.duration_seconds[item_id])
        long_threshold = min(10.0, duration)
        hlt_threshold = min(30.0, duration)
        p_long = p_play * 0.5 * erfc(
            (log(max(long_threshold, 1e-6)) - stay_log_mean) / (0.65 * sqrt(2.0))
        )
        p_hlt = p_play * 0.5 * erfc(
            (log(max(hlt_threshold, 1e-6)) - stay_log_mean) / (0.65 * sqrt(2.0))
        )
        p_like = _sigmoid(-4.2 + 1.8 * affinity + 0.8 * quality)
        p_comment = _sigmoid(-5.5 + 1.1 * affinity + 0.6 * quality)
        p_share = _sigmoid(-5.7 + 1.2 * affinity + 0.8 * quality)
        return {
            "play": p_play,
            "long_view": p_long,
            "high_quality_long_view": p_hlt,
            "stay_log_mean": stay_log_mean,
            "like": p_like,
            "favorite": p_favorite,
            "comment": p_comment,
            "share": p_share,
            "anchor_click": p_anchor,
            "poi_detail": p_detail,
            "order": p_order,
            "negative": p_negative,
        }

    def _nonlinear_adjustments(
        self, features: np.ndarray
    ) -> tuple[float, float, float]:
        if self.config.signal_version == "industrial-cross-sequence-v1":
            return 0.0, 0.0, 0.0
        estimated_affinity = float(features[0])
        quality = float(features[1])
        short_match = float(features[4])
        long_match = float(features[11])
        user_segment = int(round(float(features[14]) * 1023.0)) % 7
        item_segment = int(round(float(features[15]) * 4095.0)) % 7
        segment_match = 1.0 - abs(user_segment - item_segment) / 6.0
        threshold_cross = float(estimated_affinity > 0.28 and quality > 0.62)
        sequence_novelty = short_match * (1.0 - long_match)
        periodic = float(np.sin(np.pi * features[17] * (1.0 + features[10])))
        stay = (
            0.75 * np.tanh(2.4 * estimated_affinity * quality)
            + 0.50 * threshold_cross
            + 0.35 * segment_match
            + 0.45 * sequence_novelty
            + 0.25 * periodic
            - 1.45
        )
        local = (
            0.75 * features[13] * features[23]
            + 0.55 * features[18] * features[20]
            + 0.40 * features[19] * features[21]
            - 0.20
        )
        negative = 0.45 * float(features[7] > 0.45 and estimated_affinity < 0.15)
        return float(stay), float(local), negative

    def _response(self, features: np.ndarray, item_id: int) -> Response:
        probability = self._behavior_probabilities(features, item_id)
        rng = self._random(2)
        random = rng.random(12)
        play = bool(random[0] < probability["play"])
        duration = float(self.catalog.duration_seconds[item_id])
        stay = 0.0
        if play:
            stay = min(
                duration,
                float(rng.lognormal(probability["stay_log_mean"], 0.65)),
            )
        play_3s = play and stay >= 3.0
        slide = play and stay < 3.0
        long_view = play and stay >= min(10.0, duration)
        high_quality = play and stay >= min(30.0, duration)
        like = bool(play and random[1] < probability["like"])
        favorite = bool(play and random[2] < probability["favorite"])
        comment = bool(play and random[3] < probability["comment"])
        share = bool(play and random[4] < probability["share"])
        anchor_impression = bool(self.catalog.is_poi_video[item_id])
        anchor = bool(anchor_impression and random[5] < probability["anchor_click"])
        detail = bool(anchor and random[6] < probability["poi_detail"])
        poi_favorite = bool(detail and random[7] < probability["favorite"])
        order = bool(detail and random[8] < probability["order"])
        fulfillment = int(self.catalog.fulfillment[item_id])
        payment = bool(order and fulfillment == 1 and random[9] < 0.92)
        pixel = bool(order and fulfillment == 2 and random[10] < 0.35)
        negative = bool(random[11] < probability["negative"])
        return Response(
            play,
            play_3s,
            stay,
            slide,
            long_view,
            high_quality,
            like,
            favorite,
            comment,
            share,
            anchor_impression,
            anchor,
            detail,
            poi_favorite,
            order,
            payment,
            pixel,
            negative,
            stay / 60.0,
            probability,
        )

    def step(self, action: int):
        observation = self._observation()
        features = observation[int(action)]
        item_id = int(self.candidates[int(action)])
        response = self._response(features, item_id)
        topic = int(self.catalog.category[item_id])
        self.recent_topics.append(topic)
        self.recent_item_ids.append(item_id)
        if response.anchor_click:
            self.recent_poi_ids.append(int(self.catalog.poi[item_id]))
        engagement = (
            float(response.long_view)
            + float(response.like)
            + float(response.favorite)
            + float(response.anchor_click)
        )
        self.satisfaction = float(
            np.clip(0.82 * self.satisfaction + 0.10 * engagement - 0.24 * response.negative_feedback, -1.0, 1.0)
        )
        repeated = sum(value == topic for value in self.recent_topics)
        self.fatigue = float(np.clip(0.72 * self.fatigue + 0.08 * repeated, 0.0, 1.0))
        self.observed_satisfaction = float(
            np.clip(0.88 * self.observed_satisfaction + 0.12 * engagement - 0.20 * response.negative_feedback, -1.0, 1.0)
        )
        self.observed_fatigue = float(
            np.clip(0.82 * self.observed_fatigue + 0.05 * repeated, 0.0, 1.0)
        )
        if response.long_view:
            self.interest = 0.90 * self.interest + 0.10 * self.catalog.topics[item_id]
            self.interest /= np.linalg.norm(self.interest)
            self.observed_interest = 0.94 * self.observed_interest + 0.06 * self.catalog.topics[item_id]
            self.observed_interest /= np.linalg.norm(self.observed_interest)
            self.local_observed_interest = (
                0.94 * self.local_observed_interest
                + 0.06 * self.catalog.topics[item_id]
            )
            self.local_observed_interest /= np.linalg.norm(
                self.local_observed_interest
            )
        self.request_index += 1
        self.search_strength *= 0.78
        leave_probability = _sigmoid(-3.4 - 1.2 * self.satisfaction + 1.7 * self.fatigue)
        session_end = (
            self.request_index >= self.config.requests_per_session
            or self._random(3).random() < leave_probability
        )
        returned = False
        terminated = False
        if session_end:
            return_probability = _sigmoid(1.0 + 1.6 * self.satisfaction - 1.1 * self.fatigue + 0.4 * self.trust)
            has_next_session = self.session_id + 1 < self.config.max_sessions
            returned = bool(
                has_next_session and self._random(4).random() < return_probability
            )
            terminated = not returned
            if not terminated:
                self.session_id += 1
                self.request_index = 0
                self.fatigue *= 0.25
                self.satisfaction *= 0.75
                self.observed_fatigue *= 0.35
                self.observed_satisfaction *= 0.80
                self.interest = 0.85 * self.interest + 0.15 * self.base_interest
                self.interest /= np.linalg.norm(self.interest)
        if not terminated:
            self._refresh_candidates()
            next_observation = self._observation()
        else:
            next_observation = np.zeros_like(observation)
        info = {
            "item_id": item_id,
            "features": features,
            "response": response,
            "session_end": session_end,
            "returned": returned,
            "session_id": self.session_id,
            "request_index": self.request_index,
        }
        return next_observation, response.watch_minutes, terminated, False, info
