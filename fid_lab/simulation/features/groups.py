"""Single authority for base features and atomic feature proposals."""

from __future__ import annotations

from itertools import combinations

from ..environment import FEATURE_NAMES


BASE_FEATURE_COLUMNS = (0, 1, 2, 3, 5, 8, 9)

FEATURE_PROPOSAL_COLUMNS = {
    "sequence": (4, 11),
    "realtime": (6, 7, 10),
    "local_context": (13, 18, 19, 20, 21, 22, 23),
    "hash_content": (12, 14, 15, 16, 17),
}


def feature_set_key(proposals: tuple[str, ...]) -> str:
    unknown = set(proposals) - set(FEATURE_PROPOSAL_COLUMNS)
    if unknown:
        raise ValueError(f"unknown feature proposals: {sorted(unknown)}")
    ordered = tuple(name for name in FEATURE_PROPOSAL_COLUMNS if name in proposals)
    return "basic" if not ordered else "basic__" + "__".join(ordered)


def feature_set_columns(proposals: tuple[str, ...]) -> tuple[int, ...]:
    feature_set_key(proposals)
    columns = list(BASE_FEATURE_COLUMNS)
    for name in FEATURE_PROPOSAL_COLUMNS:
        if name in proposals:
            columns.extend(FEATURE_PROPOSAL_COLUMNS[name])
    return tuple(columns)


def feature_candidate_sets() -> dict[str, tuple[int, ...]]:
    """Materialize every legal state so promotion never triggers retraining."""
    names = tuple(FEATURE_PROPOSAL_COLUMNS)
    candidates = {}
    for size in range(len(names) + 1):
        for enabled in combinations(names, size):
            candidates[feature_set_key(enabled)] = feature_set_columns(enabled)
    return candidates


FEATURE_GROUP_COLUMNS = {
    "basic": feature_set_columns(()),
    "sequence": feature_set_columns(("sequence",)),
    "realtime": feature_set_columns(("sequence", "realtime")),
    "local_context": feature_set_columns(
        ("sequence", "realtime", "local_context")
    ),
    "full": feature_set_columns(tuple(FEATURE_PROPOSAL_COLUMNS)),
}


def feature_group_manifest() -> dict[str, object]:
    return {
        "base": tuple(FEATURE_NAMES[index] for index in BASE_FEATURE_COLUMNS),
        "proposals": {
            name: tuple(FEATURE_NAMES[index] for index in columns)
            for name, columns in FEATURE_PROPOSAL_COLUMNS.items()
        },
        "candidate_sets": {
            name: tuple(FEATURE_NAMES[index] for index in columns)
            for name, columns in feature_candidate_sets().items()
        },
    }
