"""Read-only calibration profile for the public KuaiRand-Pure log subset."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean, median


LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
FEATURE_FILES = ("user_features_pure.csv", "video_features_basic_pure.csv")
EXPECTED_SHA256 = {
    LOG_FILES[0]: "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
    LOG_FILES[1]: "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
    FEATURE_FILES[0]: "dc729a656301b4c6d07f713fe41d05ec9bfaab670b90e531c70037caf033c011",
    FEATURE_FILES[1]: "a6f7ee02684c5777422306cdc416e170302288aa89aca9dfea995edbd625bcc2",
}
FEEDBACK_FIELDS = (
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "long_view", "is_profile_enter", "is_rand",
)
REQUIRED_LOG_FIELDS = {
    "user_id", "video_id", "date", "time_ms", "play_time_ms",
    "duration_ms", "profile_stay_time", "comment_stay_time", "tab",
    *FEEDBACK_FIELDS,
}


def _file_evidence(path: Path) -> dict[str, object]:
    digest = sha256(path.read_bytes()).hexdigest()
    expected = EXPECTED_SHA256[path.name]
    if digest != expected:
        raise ValueError(f"official KuaiSim snapshot hash mismatch: {path.name}")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}


def _new_stats() -> dict[str, object]:
    return {
        "rows": 0, "users": set(), "items": set(), "feedback": Counter(),
        "play_ms": 0, "duration_ms": 0, "profile_stay_ms": 0,
        "comment_stay_ms": 0, "tabs": Counter(), "dates": [],
        "play_thresholds": Counter(),
        "last_input_time": {}, "timestamps": defaultdict(list),
        "timestamp_order_violations": 0,
    }


def _consume_log(path: Path, stats: dict[str, object]) -> set[int]:
    period_users: set[int] = set()
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_LOG_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} missing fields: {sorted(missing)}")
        for row in reader:
            user = int(row["user_id"])
            timestamp = int(row["time_ms"])
            stats["rows"] += 1
            stats["users"].add(user)
            stats["items"].add(int(row["video_id"]))
            period_users.add(user)
            for field in FEEDBACK_FIELDS:
                stats["feedback"][field] += int(row[field])
            play_ms = int(float(row["play_time_ms"]))
            duration_ms = int(float(row["duration_ms"]))
            stats["play_ms"] += play_ms
            stats["duration_ms"] += duration_ms
            stats["play_thresholds"]["positive"] += play_ms > 0
            stats["play_thresholds"]["three_seconds"] += play_ms >= 3_000
            stats["play_thresholds"]["seven_seconds"] += play_ms >= 7_000
            stats["play_thresholds"]["complete"] += play_ms >= duration_ms
            stats["profile_stay_ms"] += int(float(row["profile_stay_time"]))
            stats["comment_stay_ms"] += int(float(row["comment_stay_time"]))
            stats["tabs"][row["tab"]] += 1
            stats["dates"].append(int(row["date"]))
            previous = stats["last_input_time"].get(user)
            if previous is not None and timestamp < previous:
                stats["timestamp_order_violations"] += 1
            stats["last_input_time"][user] = timestamp
            stats["timestamps"][user].append(timestamp)
    return period_users


def _session_lengths(timestamps: dict[int, list[int]]) -> list[int]:
    lengths = []
    gap = 30 * 60 * 1000
    for values in timestamps.values():
        values.sort()
        depth = 1
        for previous, current in zip(values, values[1:]):
            if current - previous > gap:
                lengths.append(depth)
                depth = 1
            else:
                depth += 1
        lengths.append(depth)
    return lengths


def _render(stats: dict[str, object], period_users: list[set[int]]) -> dict[str, object]:
    rows = max(stats["rows"], 1)
    sessions = _session_lengths(stats["timestamps"])
    rates = {field: stats["feedback"][field] / rows for field in FEEDBACK_FIELDS}
    return {
        "interactions": stats["rows"],
        "users": len(stats["users"]),
        "items": len(stats["items"]),
        "date_range": [min(stats["dates"]), max(stats["dates"])],
        "feedback_rate": rates,
        "mean_play_seconds": stats["play_ms"] / rows / 1000.0,
        "mean_duration_seconds": stats["duration_ms"] / rows / 1000.0,
        "mean_profile_stay_seconds": stats["profile_stay_ms"] / rows / 1000.0,
        "mean_comment_stay_seconds": stats["comment_stay_ms"] / rows / 1000.0,
        "play_threshold_rate": {
            key: value / rows for key, value in stats["play_thresholds"].items()
        },
        "session_gap_minutes": 30,
        "sessions": len(sessions),
        "mean_requests_per_session": mean(sessions),
        "median_requests_per_session": median(sessions),
        "cross_period_user_return_rate": len(period_users[0] & period_users[1])
        / max(len(period_users[0]), 1),
        "source_order_violation_rate": stats["timestamp_order_violations"] / rows,
        "sessionization_order": "sort by user_id,time_ms before 30-minute split",
        "tab_share": {key: value / rows for key, value in sorted(stats["tabs"].items())},
    }


def build_kuairand_calibration(data_dir: Path) -> dict[str, object]:
    """Validate the official snapshot and render a bounded calibration profile."""
    paths = [data_dir / name for name in (*LOG_FILES, *FEATURE_FILES)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"KuaiRand calibration files missing: {missing}")
    evidence = [_file_evidence(path) for path in paths]
    stats = _new_stats()
    period_users = [_consume_log(data_dir / name, stats) for name in LOG_FILES]
    return {
        "schema": "kuairand-standard-calibration-v1",
        "source": "Applied-Machine-Learning-Lab/KuaiSim",
        "upstream_dataset": "KuaiRand-Pure",
        "license": "CC-BY-SA-4.0",
        "input_evidence": evidence,
        "profile": _render(stats, period_users),
        "causal_boundary": (
            "The available official KuaiSim subset contains standard-policy logs, "
            "not log_random; it calibrates behavior but cannot provide unbiased OPE."
        ),
    }
