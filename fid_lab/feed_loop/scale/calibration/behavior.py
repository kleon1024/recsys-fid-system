"""Versioned immediate-response parameters owned by the simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseParameters:
    play_intercept: float
    stay_intercept: float
    stay_noise: float
    like_intercept: float
    negative_intercept: float


LEGACY_RESPONSE = ResponseParameters(3.0, 0.45, 0.65, -4.2, -5.0)
KUAI_STANDARD_V1 = ResponseParameters(1.7, 0.8, 2.8, -5.5, -6.1)


def response_parameters(signal_version: str) -> ResponseParameters:
    if signal_version == "kuairand-calibrated-v3":
        return KUAI_STANDARD_V1
    return LEGACY_RESPONSE
