"""Versioned observable value tree used by the initial Feed ranker."""

from __future__ import annotations


FEED_VALUE_TREE_VERSION = "feed-engagement-v2-stay"

FEED_TASK_VALUE_WEIGHTS = {
    "play_3s": 0.15,
    "stay_value": 0.30,
    "long_view": 0.35,
    "complete": 0.15,
    "like": 0.08,
    "comment": 0.03,
    "share": 0.05,
    "follow": 0.08,
    "negative": -0.35,
}

POSTING_TASK_VALUE_WEIGHTS = {
    "click": 0.10,
    "create": 0.35,
    "publish": 0.55,
}


def task_value_weights(task_names: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(FEED_TASK_VALUE_WEIGHTS.get(name, 0.0) for name in task_names)


def surface_task_value_weights(
    task_names: tuple[str, ...],
) -> dict[int, tuple[float, ...]]:
    return {
        5: tuple(POSTING_TASK_VALUE_WEIGHTS.get(name, 0.0) for name in task_names),
    }
