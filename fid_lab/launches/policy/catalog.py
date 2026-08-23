"""Curated independent main-Feed launch candidates."""

from __future__ import annotations

from ...feed_loop.scale.tensor_engine import TensorPolicy
from ..contracts import LaunchCategory, PolicyLaunchSpec


def _personalized(name: str, **changes) -> TensorPolicy:
    values = {
        "name": name,
        "affinity_weight": 1.0,
        "quality_weight": 0.45,
        "freshness_weight": 0.10,
        "fatigue_match_penalty": 0.12,
        "eligible_fraction": 0.01,
        "observation_noise": 0.12,
        "realtime_interest_rate": 0.06,
    }
    values.update(changes)
    return TensorPolicy(**values)


def policy_launches() -> tuple[PolicyLaunchSpec, ...]:
    return (
        PolicyLaunchSpec(
            "L-FEATURE-001",
            LaunchCategory.FEATURE,
            "Lower-noise sequence interest feature",
            "A better point-in-time interest estimate improves scoped ranking.",
            "Reduce observable interest noise from 0.12 to 0.08.",
            "feed-feature",
            _personalized("feature_v1"),
            _personalized("feature_v2", observation_noise=0.08),
            "stay_per_exposure",
            "reuse_control_model_to_isolate_feature_effect",
        ),
        PolicyLaunchSpec(
            "L-STRATEGY-001",
            LaunchCategory.STRATEGY,
            "Fatigue-aware ranking constraint",
            "Penalizing repeated affinity under fatigue protects HLT.",
            "Enable a 0.12 fatigue-match penalty.",
            "feed-strategy",
            _personalized("strategy_v1", fatigue_match_penalty=0.0),
            _personalized("strategy_v2", fatigue_match_penalty=0.12),
            "quality_long_view_rate",
            "no_weight_update_strategy_only",
        ),
        PolicyLaunchSpec(
            "L-REALTIME-001",
            LaunchCategory.REALTIME,
            "Faster online interest refresh",
            "A fresher sequence state reacts faster without harming HLT.",
            "Increase online interest update rate from 0.06 to 0.12.",
            "feed-realtime",
            _personalized("realtime_v1"),
            _personalized("realtime_v2", realtime_interest_rate=0.12),
            "stay_per_exposure",
            "reuse_control_model_to_isolate_freshness_effect",
        ),
        PolicyLaunchSpec(
            "L-PRODUCT-001",
            LaunchCategory.PRODUCT,
            "Expand personalized Feed trigger",
            "Product eligibility expansion converts model value into overall ITT.",
            "Increase eligible trigger coverage from 0.5% to 1.0%.",
            "feed-product",
            _personalized("trigger_0_5pct", eligible_fraction=0.005),
            _personalized("trigger_1pct", eligible_fraction=0.01),
            "stay_per_exposure",
            "no_weight_update_trigger_only",
            product_dependency="Feed trigger eligibility and exposure logging",
        ),
        PolicyLaunchSpec(
            "L-VALUE-001",
            LaunchCategory.BUSINESS_VALUE,
            "Balanced engagement Value Tree",
            "More quality weight improves durable value without losing stay.",
            "Shift affinity/quality weights from 1.0/0.45 to 0.85/0.60.",
            "feed-value",
            _personalized("value_v1"),
            _personalized(
                "value_v2", affinity_weight=0.85, quality_weight=0.60
            ),
            "quality_long_view_rate",
            "no_weight_update_value_tree_only",
        ),
        PolicyLaunchSpec(
            "L-LONGTERM-001",
            LaunchCategory.LONG_TERM_VALUE,
            "Stronger fatigue protection",
            "More aggressive fatigue control improves long-term quality.",
            "Increase fatigue penalty from 0.12 to 0.24.",
            "feed-longterm",
            _personalized("longterm_v1"),
            _personalized("longterm_v2", fatigue_match_penalty=0.24),
            "quality_long_view_rate",
            "no_weight_update_constraint_only",
        ),
        PolicyLaunchSpec(
            "L-CHAIN-001",
            LaunchCategory.CHAIN_DIAGNOSIS,
            "Remove cold-user UID collision score",
            "Removing unrelated hashed UID noise restores candidate quality.",
            "Remove a 0.35 random collision term from cold-user ranking.",
            "feed-consistency",
            _personalized("uid_collision_bug", uid_collision_weight=0.35),
            _personalized("uid_collision_fixed", uid_collision_weight=0.0),
            "stay_per_exposure",
            "reuse_control_model_chain_fix_only",
        ),
    )
