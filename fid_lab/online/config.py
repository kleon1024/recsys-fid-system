"""One decision table for stage limits, scoring weights, rules, and constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class StageLimits:
    vector_recall: int = 240
    popular_recall: int = 80
    fresh_recall: int = 80
    merged_recall: int = 300
    coarse_rank: int = 240
    # Keep the full coarse pool so downstream policy constraints have coverage.
    fine_rank: int = 240
    policy_pool: int = 40


@dataclass(frozen=True)
class RecallConfig:
    route_weights: Mapping[str, float] = field(
        default_factory=lambda: {"viking": 1.0, "popular": 0.35, "fresh": 0.25}
    )
    reciprocal_rank_constant: float = 20.0


@dataclass(frozen=True)
class ValueTreeConfig:
    engagement_weights: Mapping[str, float] = field(
        default_factory=lambda: {"p_click": 0.45, "p_like": 0.25, "p_long_view": 0.30}
    )
    ecosystem_weights: Mapping[str, float] = field(
        default_factory=lambda: {"quality": 0.75, "freshness": 0.25}
    )
    root_weights: Mapping[str, float] = field(
        default_factory=lambda: {"engagement": 0.72, "ecosystem": 0.28}
    )


@dataclass(frozen=True)
class RuleConfig:
    fresh_multiplier: float = 1.06
    high_quality_threshold: float = 0.82
    quality_multiplier: float = 1.04
    type_multipliers: Mapping[str, float] = field(
        default_factory=lambda: {"organic": 1.0, "live": 0.98, "ad": 0.92}
    )


@dataclass(frozen=True)
class PolicyConfig:
    min_fresh: int = 3
    max_per_creator: int = 2
    max_per_category: int = 7
    exploration_bonus: float = 0.025


@dataclass(frozen=True)
class MixConfig:
    max_by_type: Mapping[str, int] = field(
        default_factory=lambda: {"organic": 20, "live": 3, "ad": 2}
    )
    score_calibration: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "organic": (1.0, 0.0),
            "live": (0.97, 0.01),
            "ad": (0.90, 0.015),
        }
    )
    max_consecutive_category: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    version: str = "pipeline-v1"
    fresh_age_hours: float = 36.0
    limits: StageLimits = field(default_factory=StageLimits)
    recall: RecallConfig = field(default_factory=RecallConfig)
    value_tree: ValueTreeConfig = field(default_factory=ValueTreeConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    mix: MixConfig = field(default_factory=MixConfig)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()
