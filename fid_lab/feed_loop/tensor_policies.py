"""Parameter-only policies shared by tensor Feed experiment surfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorPolicy:
    name: str
    affinity_weight: float
    quality_weight: float
    freshness_weight: float
    fatigue_match_penalty: float = 0.0
    eligible_fraction: float = 1.0
    observation_noise: float = 0.12
    local_observation_noise: float = 0.12
    realtime_interest_rate: float = 0.06
    uid_collision_weight: float = 0.0
    local_weight: float = 0.0
    search_weight: float = 0.0
    retarget_weight: float = 0.0
    local_intent_quality_weight: float = 0.0
    local_embedding_correction_weight: float = 0.0
    coarse_keep: int = 0
    coarse_affinity_weight: float = 1.0
    coarse_quality_weight: float = 0.35
    coarse_local_weight: float = 0.05
    multi_queue: bool = False
    ad_weight: float = 0.0
    live_weight: float = 0.0
    max_ads_per_session: int = 0
    min_ad_gap: int = 4
    max_live_per_session: int = 0


POPULAR = TensorPolicy("quality_baseline", 0.0, 1.0, 0.15)
PERSONALIZED = TensorPolicy("personalized_rank", 1.0, 0.45, 0.10, 0.12)
PERSONALIZED_1PCT = TensorPolicy(
    "personalized_rank_1pct_trigger", 1.0, 0.45, 0.10, 0.12, 0.01
)
LOCAL_STATIC = TensorPolicy(
    "local_static_v1", 1.0, 0.45, 0.10, 0.12, local_weight=0.08
)
LOCAL_SEARCH = TensorPolicy(
    "local_post_search_v2",
    1.0,
    0.45,
    0.10,
    0.12,
    local_weight=0.08,
    search_weight=0.55,
)
LOCAL_RETARGET = TensorPolicy(
    "local_search_retarget_v3",
    1.0,
    0.45,
    0.10,
    0.12,
    local_weight=0.08,
    search_weight=0.55,
    retarget_weight=0.45,
)
LOCAL_INTENT_RANKER = TensorPolicy(
    "local_intent_quality_rank_v4",
    1.0,
    0.45,
    0.10,
    0.12,
    local_weight=0.08,
    search_weight=0.55,
    retarget_weight=0.45,
    local_intent_quality_weight=0.10,
    local_observation_noise=0.04,
    local_embedding_correction_weight=1.0,
)
LOCAL_EXPANSION = TensorPolicy(
    "local_value_expansion_v5",
    1.0,
    0.45,
    0.10,
    0.12,
    local_weight=0.14,
    search_weight=0.55,
    retarget_weight=0.45,
    local_intent_quality_weight=0.10,
    local_observation_noise=0.04,
    local_embedding_correction_weight=1.0,
)
