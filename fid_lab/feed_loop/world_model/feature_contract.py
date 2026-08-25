"""Versioned semantic contract for the NeuralSCM 28-field input."""

from __future__ import annotations

from hashlib import sha256
import json


FEATURE_CONTRACT_SCHEMA = "neural-scm-feature-contract-v1"
FEATURE_COUNT = 28
CANONICAL_FEATURE_FIELDS = (
    "history_topic_affinity", "content_quality_prior_sqrt", "unused_02",
    "content_quality_prior", "recent_topic_match", "same_region",
    "history_engagement_rate", "history_negative_rate", "user_activity",
    "user_habit", "history_fill_ratio", "long_topic_match",
    "duration_log_seconds", "exact_repeat_log", "creator_repeat_log",
    "topic_repeat_log", "user_disappointment", "content_topic",
    "same_country", "recent_topic_match_copy",
    "feed_surface", "bias", "content_age", "user_novelty", "account_age",
    "user_activity_copy", "user_lifecycle", "repeat_penalty",
)
V4_REQUIRED_FEATURES = frozenset(
    index for index, name in enumerate(CANONICAL_FEATURE_FIELDS)
    if not name.startswith("unused_")
)
SUPPORT_BOUNDED_FEATURES = frozenset({4, 5, 17, 18, 19, 20, 21, 26})


def feature_contract(fields: tuple[str, ...]) -> dict:
    if len(fields) != FEATURE_COUNT:
        raise ValueError(f"feature contract requires {FEATURE_COUNT} fields")
    if len(set(fields)) != len(fields):
        raise ValueError("feature semantics must be unique")
    payload = {
        "schema": FEATURE_CONTRACT_SCHEMA,
        "fields": [
            {"index": index, "semantic": semantic}
            for index, semantic in enumerate(fields)
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return {**payload, "sha256": sha256(encoded).hexdigest()}


def compare_feature_contracts(left: dict, right: dict) -> list[dict]:
    left_fields = left.get("fields", [])
    right_fields = right.get("fields", [])
    if len(left_fields) != len(right_fields):
        return [{
            "index": None,
            "left": f"count={len(left_fields)}",
            "right": f"count={len(right_fields)}",
        }]
    return [
        {
            "index": index,
            "left": left_row.get("semantic"),
            "right": right_row.get("semantic"),
        }
        for index, (left_row, right_row) in enumerate(
            zip(left_fields, right_fields, strict=True)
        )
        if left_row.get("semantic") != right_row.get("semantic")
    ]
