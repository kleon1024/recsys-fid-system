"""Versioned Local cascade kernels, including the hidden neural V4 teacher."""

from __future__ import annotations

from math import sqrt

import torch
from torch.nn import functional as functional

from ..graph.random import uniform


LOCAL_NEURAL_SIGNAL_VERSION = "kuairand-local-neural-v4"
LOCAL_NEURAL_INTERCEPTS = (-3.75, -0.80, -2.80, -2.40)


def _legacy_logits(
    affinity, commerce, poi_quality, inventory, same_city, search_match,
    retarget_match,
):
    return torch.stack((
        -5.0 + 1.7 * affinity + 0.7 * same_city + 0.5 * commerce
        + 1.4 * search_match + 1.1 * retarget_match,
        -1.3 + affinity + 0.8 * poi_quality + 0.7 * search_match,
        -3.2 + 0.8 * affinity + poi_quality,
        -4.5 + 1.2 * commerce + 0.9 * poi_quality
        + 0.9 * retarget_match + 1.2 * inventory,
    ), dim=1)


def _neural_logits(
    user_ids, affinity, commerce, poi_quality, inventory, same_city,
    search_match, retarget_match, fulfillment,
):
    cohort_phase = torch.remainder(user_ids, 997).float() / 997.0
    inputs = torch.stack((
        affinity,
        commerce,
        poi_quality,
        inventory,
        same_city,
        search_match,
        retarget_match,
        (fulfillment == 1).float(),
        (fulfillment == 2).float(),
        torch.sin(2.0 * torch.pi * cohort_phase),
        torch.cos(2.0 * torch.pi * cohort_phase),
    ), dim=1)
    rows = torch.arange(
        1, inputs.shape[1] + 1, device=inputs.device, dtype=inputs.dtype
    )[:, None]
    columns = torch.arange(1, 17, device=inputs.device, dtype=inputs.dtype)[None, :]
    first = torch.sin(rows * columns * 0.731) * 0.70
    bias = torch.cos(columns.squeeze(0) * 0.419) * 0.20
    hidden = functional.silu(inputs @ first + bias)
    outputs = torch.arange(2, 6, device=inputs.device, dtype=inputs.dtype)[None, :]
    second = torch.cos(columns.T * outputs * 0.377) * 0.55
    interaction = hidden @ second / sqrt(hidden.shape[1])
    structured = torch.stack((
        0.75 * affinity + 0.35 * same_city + 0.55 * search_match,
        0.45 * affinity + 0.35 * poi_quality + 0.30 * search_match,
        0.25 * affinity + 0.40 * poi_quality,
        0.35 * commerce + 0.40 * inventory + 0.30 * retarget_match,
    ), dim=1)
    intercept = torch.tensor(
        LOCAL_NEURAL_INTERCEPTS,
        device=inputs.device, dtype=inputs.dtype,
    )
    return intercept + structured + 1.25 * interaction


def sample_local_response(
    user_ids, step, seed, signal_version, active, affinity, is_poi,
    commerce, poi_quality, inventory, same_city, search_match,
    retarget_match, fulfillment,
):
    if signal_version == LOCAL_NEURAL_SIGNAL_VERSION:
        logits = _neural_logits(
            user_ids, affinity, commerce, poi_quality, inventory, same_city,
            search_match, retarget_match, fulfillment,
        )
    else:
        logits = _legacy_logits(
            affinity, commerce, poi_quality, inventory, same_city,
            search_match, retarget_match,
        )
    anchor = (
        uniform(user_ids, step, 40, seed) < torch.sigmoid(logits[:, 0])
    ) & active & is_poi.bool()
    detail = anchor & (
        uniform(user_ids, step, 41, seed) < torch.sigmoid(logits[:, 1])
    )
    favorite = detail & (
        uniform(user_ids, step, 42, seed) < torch.sigmoid(logits[:, 2])
    )
    order = detail & (
        uniform(user_ids, step, 43, seed) < torch.sigmoid(logits[:, 3])
    )
    paid = order & (fulfillment == 1) & (
        uniform(user_ids, step, 44, seed) < 0.92
    )
    pixel = order & (fulfillment == 2) & (
        uniform(user_ids, step, 45, seed) < 0.35
    )
    return anchor, detail, favorite, paid, pixel
