"""Shared score-composition contract for Feed Posting serving paths."""

from __future__ import annotations


BLEND_MODES = ("legacy_convex", "standardized_residual")


def request_standardize(values):
    return (values - values.mean(1, keepdim=True)) / (
        values.std(1, keepdim=True).clamp_min(1e-4)
    )


def blend_score(baseline, learned, blend, mode):
    if mode == "legacy_convex":
        return baseline + blend * (learned - baseline)
    if mode == "standardized_residual":
        return request_standardize(baseline) + blend * request_standardize(
            learned
        )
    raise ValueError(f"unknown Feed Posting blend mode: {mode}")


def policy_name(model, blend, mode):
    suffix = "" if mode == "legacy_convex" else f"_{mode}"
    return f"{model}{suffix}_blend_{blend:.2f}"
