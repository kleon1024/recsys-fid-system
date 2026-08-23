"""Shared calibrated policy scoring for shadow replay and randomized OPE."""

from __future__ import annotations

import torch

from ..contracts import FEEDBACK_NAMES
from ..kernel import SlateResponse
from ..launch.contracts import PolicySpec


STANDARDIZED_FEED_WEIGHTS = {
    "click": 0.25,
    "long_view": 0.40,
    "stay_norm": 0.30,
    "hate": -0.05,
}
RAW_SELECTION_WEIGHTS = {
    "click": 0.10,
    "long_view": 0.27,
    "like": 0.30,
    "hate": -0.10,
    "stay_norm": 0.28,
}


def calibrate_response(response: SlateResponse, rules: dict) -> SlateResponse:
    values = []
    for index, name in enumerate(FEEDBACK_NAMES):
        rule = rules[name]["rule"]
        probability = response.probabilities[:, :, index].clamp(1e-6, 1 - 1e-6)
        values.append(
            torch.sigmoid(
                rule["coefficient"] * torch.logit(probability) + rule["intercept"]
            )
        )
    stay_rule = rules["stay_norm"]["rule"]
    stay = (
        stay_rule["coefficient"] * response.stay_norm + stay_rule["intercept"]
    ).clamp(0.0, 1.0)
    return SlateResponse(torch.stack(values, dim=2), stay)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-4)


def policy_utility(response: SlateResponse, mode: str) -> torch.Tensor:
    if mode == "raw_probability":
        probability = response.probabilities
        return (
            RAW_SELECTION_WEIGHTS["click"] * probability[:, :, 0]
            + RAW_SELECTION_WEIGHTS["long_view"] * probability[:, :, 1]
            + RAW_SELECTION_WEIGHTS["like"] * probability[:, :, 2]
            + RAW_SELECTION_WEIGHTS["hate"] * probability[:, :, 6]
            + RAW_SELECTION_WEIGHTS["stay_norm"] * response.stay_norm
        )
    if mode != "standardized_feed":
        raise ValueError(f"unsupported policy utility: {mode}")
    return (
        STANDARDIZED_FEED_WEIGHTS["click"]
        * _standardize(response.probabilities[:, :, 0])
        + STANDARDIZED_FEED_WEIGHTS["long_view"]
        * _standardize(response.probabilities[:, :, 1])
        + STANDARDIZED_FEED_WEIGHTS["stay_norm"] * _standardize(response.stay_norm)
        + STANDARDIZED_FEED_WEIGHTS["hate"]
        * _standardize(response.probabilities[:, :, 6])
    )


def policy_distribution(
    response: SlateResponse, spec: PolicySpec, eligible: torch.Tensor
) -> torch.Tensor:
    utility = policy_utility(response, spec.utility_mode)
    utility = utility.masked_fill(~eligible[None], -torch.inf)
    learned = torch.softmax(utility / spec.temperature, dim=1)
    return spec.uniform_mixture / utility.shape[1] + (
        1.0 - spec.uniform_mixture
    ) * learned
