"""Fixed evidence-role profiles; dimensions cannot be mixed post hoc."""

from __future__ import annotations

from .contracts import TwinConfig


PROFILE_OVERRIDES = {
    "smoke": {
        "users": 10_000,
        "catalog_items": 100_000,
        "creators": 10_000,
        "preperiod_steps": 2,
        "measurement_steps": 4,
        "steps_per_day": 4,
        "history_length": 16,
        "route_candidates": 8,
        "coarse_keep": 24,
        "fine_keep": 8,
        "audit_users": 256,
        "training_trace_users": 4_096,
        "batch_users": 10_000,
        "serve_chunk_users": 10_000,
    },
    "screen": {
        "users": 100_000,
        "catalog_items": 500_000,
        "creators": 50_000,
        "preperiod_steps": 4,
        "measurement_steps": 12,
        "steps_per_day": 6,
        "history_length": 32,
        "route_candidates": 12,
        "coarse_keep": 36,
        "fine_keep": 10,
        "audit_users": 1_024,
        "training_trace_users": 16_384,
        "batch_users": 100_000,
        "serve_chunk_users": 50_000,
    },
    "gpu": {
        "users": 1_000_000,
        "catalog_items": 2_000_000,
        "creators": 250_000,
        "preperiod_steps": 8,
        "measurement_steps": 32,
        "steps_per_day": 8,
        "history_length": 64,
        "route_candidates": 16,
        "coarse_keep": 48,
        "fine_keep": 12,
        "audit_users": 2_560,
        "training_trace_users": 16_384,
        "batch_users": 250_000,
        "serve_chunk_users": 50_000,
    },
}


def load_profile(name: str, device="cuda:0") -> TwinConfig:
    if name not in PROFILE_OVERRIDES:
        raise ValueError(f"unknown digital-twin profile: {name}")
    return TwinConfig(device=device, **PROFILE_OVERRIDES[name])
