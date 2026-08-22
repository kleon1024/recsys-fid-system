"""Typed configuration authority for overlapping Feed experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class FeedParameters:
    recall_model: str = "two_tower_v1"
    enabled_routes: tuple[str, ...] = (
        "ann",
        "graph",
        "geo",
        "fresh",
        "long_tail",
        "popular",
    )
    recall_budget: int = 100
    coarse_model: str = "lr_v1"
    coarse_budget: int = 20
    fine_model: str = "lr_v1"
    calibration_temperature: float = 1.0
    stay_weight: float = 1.0
    long_view_weight: float = 1.0
    hlt_weight: float = 0.5
    interaction_weight: float = 0.25
    negative_weight: float = -1.0
    diversity_strength: float = 0.0
    freshness_boost: float = 0.0
    exploration_rate: float = 0.0
    feature_snapshot: str = "feed_features_v1"
    model_manifest: str = "feed_manifest_v1"

    def overlay(self, changes: Mapping[str, object]) -> "FeedParameters":
        unknown = set(changes) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown Feed parameters: {sorted(unknown)}")
        candidate = replace(self, **changes)
        if candidate.coarse_budget > candidate.recall_budget:
            raise ValueError("coarse_budget cannot exceed recall_budget")
        if candidate.calibration_temperature <= 0.0:
            raise ValueError("calibration_temperature must be positive")
        if not 0.0 <= candidate.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0, 1]")
        return candidate


@dataclass(frozen=True)
class Variant:
    name: str
    allocation: float
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    name: str
    variants: tuple[Variant, ...]

    def __post_init__(self) -> None:
        allocation = sum(variant.allocation for variant in self.variants)
        if not self.variants or allocation > 1.0 + 1e-12:
            raise ValueError("experiment variants must allocate at most one layer")
        if any(variant.allocation <= 0.0 for variant in self.variants):
            raise ValueError("variant allocation must be positive")


@dataclass(frozen=True)
class ExperimentLayer:
    name: str
    salt: str
    experiments: tuple[Experiment, ...]

    def __post_init__(self) -> None:
        allocation = sum(
            variant.allocation
            for experiment in self.experiments
            for variant in experiment.variants
        )
        if allocation > 1.0 + 1e-12:
            raise ValueError("experiments in one layer must be mutually exclusive")
