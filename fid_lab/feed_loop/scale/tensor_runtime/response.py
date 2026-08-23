"""Vectorized Feed and Local behavior response kernels."""

from __future__ import annotations

import torch

from ....value import BUSINESS_TREE_WEIGHTS, DEFAULT_LT_CONFIG
from ..calibration.behavior import response_parameters
from ..graph.random import normal, uniform
from .local_response import sample_local_response


def sample_response(
    user_ids, step, seed, signal_version, active, fatigue, satisfaction, affinity, quality,
    duration=None, stay_adjustment=None,
):
    parameters = response_parameters(signal_version)
    if duration is None:
        duration = 3.0 + 177.0 * uniform(user_ids, step, 30, seed)
    play_draw = (
        uniform(user_ids, step, 31, seed)
        < torch.sigmoid(
            parameters.play_intercept + 0.4 * affinity - 0.6 * fatigue
        )
    )
    played = play_draw & active
    stay_log_mean = (
        parameters.stay_intercept
        + 1.7 * affinity + 0.55 * quality + 0.20 * satisfaction - fatigue
    )
    if stay_adjustment is not None:
        stay_log_mean += stay_adjustment
    stay = torch.minimum(
        duration,
        torch.exp(
            stay_log_mean
            + parameters.stay_noise * normal(user_ids, step, 32, seed)
        ),
    ) * played
    long_threshold = 18.0 if signal_version.startswith("kuairand-") else 10.0
    long_view = (
        stay >= torch.minimum(torch.full_like(stay, long_threshold), duration)
    ) & active
    hlt = (stay >= torch.minimum(torch.full_like(stay, 30.0), duration)) & active
    like = (
        uniform(user_ids, step, 34, seed)
        < torch.sigmoid(parameters.like_intercept + 1.8 * affinity + 0.8 * quality)
    ) & played
    negative = (
        uniform(user_ids, step, 35, seed)
        < torch.sigmoid(
            parameters.negative_intercept
            - 1.7 * affinity - 0.8 * quality + 2.0 * fatigue
        )
    ) & active
    return stay, long_view, hlt, like, negative, played, play_draw


def business_and_lt_values(
    stay,
    hlt,
    like,
    negative,
    anchor,
    detail,
    favorite,
    paid,
    pixel,
    commerce,
):
    tree = BUSINESS_TREE_WEIGHTS
    lt_rates = DEFAULT_LT_CONFIG.rates
    feed_tree = (
        hlt.float() * tree["quality_view"]
        + like.float() * tree["meaningful_engagement"]
        + negative.float() * tree["negative_feedback"]
    )
    converted = paid | pixel
    terminal_intent = torch.where(
        converted,
        torch.zeros_like(stay),
        torch.where(
            favorite,
            torch.full_like(stay, tree["poi_favorite"]),
            torch.where(
                detail,
                torch.full_like(stay, tree["poi_detail"]),
                anchor.float() * tree["anchor_click"],
            ),
        ),
    )
    transaction = (
        paid.float() * tree["closed_loop_payment"]
        + pixel.float() * tree["open_loop_conversion"]
        + converted.float() * commerce * 8.0 * tree["contribution_margin"]
    )
    local_tree = terminal_intent + transaction
    commercialization = converted.float() * commerce * 8.0
    lt_value = stay / 60.0 * lt_rates["stay_minute"].unit_value
    return lt_value, feed_tree, local_tree, commercialization
