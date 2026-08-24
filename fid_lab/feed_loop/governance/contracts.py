"""Versioned authority for fast, reversible Feed governance policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContentGovernanceConfig:
    version: str = "content-governance-v1"
    max_predicted_integrity_risk: float = 0.75
    repeated_cluster_penalty: float = 0.02
    repeated_author_penalty: float = 0.01
    new_creator_boost: float = 0.0
    max_poi_per_session: int = 1_000_000
    min_poi_gap: int = 0

    def __post_init__(self):
        if not 0.0 < self.max_predicted_integrity_risk < 1.0:
            raise ValueError("integrity threshold must be a probability")
        if min(
            self.repeated_cluster_penalty,
            self.repeated_author_penalty,
            self.new_creator_boost,
        ) < 0.0:
            raise ValueError("governance score adjustments must be nonnegative")
        if self.max_poi_per_session < 0 or self.min_poi_gap < 0:
            raise ValueError("POI governance limits must be nonnegative")

    def manifest(self):
        return asdict(self)


@dataclass(frozen=True)
class GovernanceLaunchThresholds:
    shadow_lt_noninferiority: float = -0.001
    shadow_stay_noninferiority: float = -0.02
    shadow_quality_view_noninferiority: float = -0.001
    shadow_negative_upper: float = 0.0002
    shadow_duplicate_upper: float = 0.0005
    online_lt_noninferiority: float = -0.01
    online_stay_noninferiority: float = -0.02
    online_quality_view_noninferiority: float = -0.002
    online_negative_upper: float = 0.0002

    def manifest(self):
        return asdict(self)
