"""Ordered feature groups for isolated logistic-regression launches."""

from __future__ import annotations

from ..environment import FEATURE_NAMES


FEATURE_GROUP_COLUMNS = {
    "basic": (0, 1, 2, 3, 5, 8, 9),
    "sequence": (0, 1, 2, 3, 5, 8, 9, 4, 11),
    "realtime": (0, 1, 2, 3, 5, 8, 9, 4, 11, 6, 7, 10),
    "local_context": (
        0, 1, 2, 3, 5, 8, 9, 4, 11, 6, 7, 10, 13, 18, 19, 20, 21, 22, 23,
    ),
    "full": tuple(range(len(FEATURE_NAMES))),
}


def feature_group_manifest() -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(FEATURE_NAMES[index] for index in columns)
        for name, columns in FEATURE_GROUP_COLUMNS.items()
    }
