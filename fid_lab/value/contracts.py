"""Authorities for business Value Trees and the platform LT metric container.

Business Value Trees produce ranking and diagnostic scores. They are not LT
units. LT consumes only measured platform metrics: stay, active days, and a
platform-accepted commercialization measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ExchangeRate:
    unit_value: float
    standard_error: float
    evidence: str

    def __post_init__(self) -> None:
        if self.standard_error < 0.0:
            raise ValueError("exchange-rate standard error cannot be negative")


@dataclass(frozen=True)
class LTMetricConfig:
    version: str
    rates: Mapping[str, ExchangeRate] = field(default_factory=dict)


@dataclass(frozen=True)
class LTMetricVector:
    stay_minutes: float = 0.0
    active_days: float = 0.0
    accepted_commercialization_value: float = 0.0


@dataclass(frozen=True)
class LTMetricBreakdown:
    version: str
    stay: float
    active_days: float
    accepted_commercialization: float
    total: float


@dataclass(frozen=True)
class BusinessValueSignals:
    quality_view: int = 0
    meaningful_engagements: int = 0
    negative_feedback: int = 0
    anchor_click: int = 0
    poi_detail: int = 0
    poi_favorite: int = 0
    closed_loop_payment: int = 0
    open_loop_verified_conversion: float = 0.0
    contribution_margin: float = 0.0
    quality_adjusted_supply: float = 0.0
    ad_value: float = 0.0
    live_value: float = 0.0


@dataclass(frozen=True)
class BusinessValueBreakdown:
    version: str
    feed: float
    local_consumption: float
    local_transaction: float
    local_supply: float
    ads_live: float


def _synthetic_rate(value: float, standard_error: float = 0.0) -> ExchangeRate:
    return ExchangeRate(value, standard_error, "synthetic_platform_metric_calibration_v1")


DEFAULT_LT_CONFIG = LTMetricConfig(
    version="lt-platform-metrics-v1",
    rates={
        "stay_minute": _synthetic_rate(1.0, 0.03),
        "active_day": _synthetic_rate(5.0, 0.30),
        "accepted_commercialization_unit": _synthetic_rate(1.0, 0.08),
    },
)


BUSINESS_TREE_WEIGHTS = {
    "quality_view": 1.0,
    "meaningful_engagement": 1.25,
    "negative_feedback": -4.0,
    "anchor_click": 0.75,
    "poi_detail": 1.75,
    "poi_favorite": 3.0,
    "closed_loop_payment": 6.0,
    "open_loop_conversion": 5.0,
    "contribution_margin": 1.0,
    "quality_adjusted_supply": 3.5,
    "ad_value": 1.0,
    "live_value": 1.0,
}
