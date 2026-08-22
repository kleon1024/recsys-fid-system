"""Gymnasium environment with request, session, and cross-session state."""

from __future__ import annotations

from collections import deque
from math import exp

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .contracts import Catalog, Response, SimulationConfig


FEATURE_NAMES = (
    "interest_affinity",
    "item_quality",
    "commerce_value",
    "topic_novelty",
    "same_city",
    "satisfaction",
    "fatigue",
    "trust",
    "commerce_propensity",
    "session_progress",
)


def build_catalog(config: SimulationConfig) -> Catalog:
    rng = np.random.default_rng(config.seed)
    topics = rng.gamma(0.7, 1.0, size=(config.items, config.topics)).astype(np.float32)
    topics /= np.maximum(np.linalg.norm(topics, axis=1, keepdims=True), 1e-8)
    return Catalog(
        topics=topics,
        quality=rng.beta(3.0, 2.2, config.items).astype(np.float32),
        commerce_value=rng.beta(1.4, 5.0, config.items).astype(np.float32),
        category=np.argmax(topics, axis=1).astype(np.int32),
        city=rng.integers(0, 100, config.items, dtype=np.int32),
        author=rng.integers(0, max(config.items // 8, 1), config.items, dtype=np.int32),
        poi=rng.integers(0, max(config.items // 4, 1), config.items, dtype=np.int32),
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-max(min(value, 20.0), -20.0)))


class StatefulFeedEnv(gym.Env):
    """POI Feed dynamics on the standard Gymnasium environment contract."""

    metadata = {"render_modes": []}

    def __init__(self, config: SimulationConfig, catalog: Catalog) -> None:
        super().__init__()
        self.config = config
        self.catalog = catalog
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
        self.satisfaction = float(user_rng.uniform(-0.05, 0.05))
        self.fatigue = 0.0
        self.trust = float(user_rng.beta(5.0, 2.0))
        self.commerce_propensity = float(user_rng.beta(1.5, 5.0))
        self.city = int(user_rng.integers(0, 100))
        self.session_id = 0
        self.request_index = 0
        self.recent_topics: deque[int] = deque(maxlen=8)
        self.candidates = self._candidate_ids()
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

    def _candidate_ids(self) -> np.ndarray:
        return self._random(1).choice(
            self.config.items, self.config.candidates, replace=False
        )

    def _observation(self) -> np.ndarray:
        item_topics = self.catalog.topics[self.candidates]
        affinity = item_topics @ self.interest
        categories = self.catalog.category[self.candidates]
        recent = np.fromiter(self.recent_topics, dtype=np.int32)
        novelty = (
            np.ones(len(categories), dtype=np.float32)
            if not len(recent)
            else (~np.isin(categories, recent)).astype(np.float32)
        )
        return np.column_stack(
            (
                affinity,
                self.catalog.quality[self.candidates],
                self.catalog.commerce_value[self.candidates],
                novelty,
                (self.catalog.city[self.candidates] == self.city).astype(np.float32),
                np.full(self.config.candidates, self.satisfaction, dtype=np.float32),
                np.full(self.config.candidates, self.fatigue, dtype=np.float32),
                np.full(self.config.candidates, self.trust, dtype=np.float32),
                np.full(
                    self.config.candidates, self.commerce_propensity, dtype=np.float32
                ),
                np.full(
                    self.config.candidates,
                    self.request_index / self.config.requests_per_session,
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)

    def _response(self, features: np.ndarray) -> Response:
        affinity, quality, value, novelty, same_city, satisfaction, fatigue, trust, commerce, _ = features
        nonlinear_match = float(affinity > 0.55 and quality > 0.65)
        p_long = _sigmoid(
            -5.0
            + 3.0 * affinity
            + 0.8 * quality
            + 2.0 * affinity * quality
            + 0.8 * same_city * affinity
            + 0.7 * nonlinear_match
            + 0.4 * novelty
            + satisfaction
            - 1.5 * fatigue * (1.0 - 0.7 * novelty)
        )
        p_anchor = _sigmoid(-5.0 + 2.1 * affinity + 1.0 * same_city + 0.7 * trust + 0.5 * value)
        p_detail = _sigmoid(-1.1 + 1.3 * affinity + 1.0 * quality + 0.7 * same_city)
        p_favorite = _sigmoid(-2.5 + 1.8 * affinity + 0.8 * quality)
        p_order = _sigmoid(-4.3 + 1.4 * value + 1.0 * commerce + 0.6 * trust)
        p_negative = _sigmoid(-5.0 - 1.7 * affinity - 0.8 * quality + 2.0 * fatigue - satisfaction)
        random = self._random(2).random(8)
        long_view = bool(random[0] < p_long)
        anchor = bool(random[1] < p_anchor)
        detail = bool(anchor and random[2] < p_detail)
        favorite = bool(detail and random[3] < p_favorite)
        order = bool(detail and random[4] < p_order)
        payment = bool(order and random[5] < 0.92)
        pixel = bool(payment and random[6] < 0.35)
        negative = bool(random[7] < p_negative)
        watch = (0.15 + 2.8 * affinity + 1.2 * quality) * (0.25 + 0.75 * long_view)
        return Response(
            long_view,
            anchor,
            detail,
            favorite,
            order,
            payment,
            pixel,
            negative,
            max(float(watch), 0.0),
            (p_long, p_anchor, p_detail, p_favorite, p_order, p_negative),
        )

    def step(self, action: int):
        observation = self._observation()
        features = observation[int(action)]
        item_id = int(self.candidates[int(action)])
        response = self._response(features)
        topic = int(self.catalog.category[item_id])
        self.recent_topics.append(topic)
        engagement = float(response.long_view) + float(response.anchor_click) + float(response.favorite)
        self.satisfaction = float(
            np.clip(0.82 * self.satisfaction + 0.10 * engagement - 0.24 * response.negative_feedback, -1.0, 1.0)
        )
        repeated = sum(value == topic for value in self.recent_topics)
        self.fatigue = float(np.clip(0.72 * self.fatigue + 0.08 * repeated, 0.0, 1.0))
        if response.long_view:
            self.interest = 0.90 * self.interest + 0.10 * self.catalog.topics[item_id]
            self.interest /= np.linalg.norm(self.interest)
        self.request_index += 1
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
                self.interest = 0.85 * self.interest + 0.15 * self.base_interest
                self.interest /= np.linalg.norm(self.interest)
        if not terminated:
            self.candidates = self._candidate_ids()
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
