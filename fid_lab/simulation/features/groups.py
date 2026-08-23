"""Single authority for base features and atomic feature proposals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..contracts import DEFAULT_SEARCH_EVENT_RATE
from ..environment import FEATURE_NAMES


BASE_FEATURE_COLUMNS = (0, 1, 2, 3, 5, 8, 9)

FEATURE_PROPOSAL_COLUMNS = {
    "sequence": (4, 11),
    "realtime": (6, 7, 10),
    "local_context": (13, 18, 19, 20, 21, 22, 23),
    "hash_content": (12, 14, 15, 16, 17),
}


@dataclass(frozen=True)
class FeatureCampaign:
    name: str
    base_key: str
    base_columns: tuple[int, ...]
    proposals: tuple[tuple[str, tuple[int, ...]], ...]
    launch_start: int
    report_logical_key: str
    artifact_collection: str
    operation: str
    trigger_kinds: tuple[tuple[str, str], ...] = ()
    burn_in_steps: int = 0
    search_event_rate: float = DEFAULT_SEARCH_EVENT_RATE


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


FEATURE_CAMPAIGNS = {
    "hash_content_split_v1": FeatureCampaign(
        name="hash_content_split_v1",
        base_key="basic__realtime__local_context",
        base_columns=feature_set_columns(("realtime", "local_context")),
        proposals=(
            ("duration", (12,)),
            ("identity_hash", (14, 15, 16)),
            ("category_hash", (17,)),
        ),
        launch_start=5,
        report_logical_key="feature-lr-hash-content-split-ab",
        artifact_collection="feature-lr-v3-hash-split",
        operation="add",
    ),
    "local_ablation_v1": FeatureCampaign(
        name="local_ablation_v1",
        base_key="basic__realtime__local_context__category_hash",
        base_columns=(
            *feature_set_columns(("realtime", "local_context")),
            17,
        ),
        proposals=(
            ("without_poi_indicator", (13,)),
            ("without_post_search", (18,)),
            ("without_retarget", (19,)),
            ("without_quality_inventory", (20, 21)),
            ("without_geo_interest", (22, 23)),
        ),
        launch_start=8,
        report_logical_key="feature-lr-local-ablation-ab",
        artifact_collection="feature-lr-v4-local-ablation",
        operation="remove",
    ),
    "intent_trigger_ablation_v1": FeatureCampaign(
        name="intent_trigger_ablation_v1",
        base_key="basic__realtime__local_context__category_hash",
        base_columns=(
            *feature_set_columns(("realtime", "local_context")),
            17,
        ),
        proposals=(
            ("without_post_search", (18,)),
            ("without_retarget", (19,)),
        ),
        launch_start=13,
        report_logical_key="feature-lr-intent-trigger-ablation-ab",
        artifact_collection="feature-lr-v5-intent-trigger",
        operation="remove",
        trigger_kinds=(
            ("without_post_search", "post_search"),
            ("without_retarget", "retarget"),
        ),
        burn_in_steps=8,
        search_event_rate=0.04,
    ),
}


def feature_campaign(name: str) -> FeatureCampaign:
    try:
        return FEATURE_CAMPAIGNS[name]
    except KeyError as error:
        raise ValueError(f"unknown feature campaign: {name}") from error


def _campaign_columns(campaign, proposals, enabled):
    changed = tuple(
        column for proposal in enabled for column in proposals[proposal]
    )
    if campaign.operation == "add":
        return campaign.base_columns + changed
    if campaign.operation == "remove":
        return tuple(
            column for column in campaign.base_columns if column not in changed
        )
    raise ValueError(f"unsupported campaign operation: {campaign.operation}")


def campaign_candidate_sets(name: str) -> dict[str, tuple[int, ...]]:
    campaign = feature_campaign(name)
    proposals = dict(campaign.proposals)
    candidates = {}
    for size in range(len(proposals) + 1):
        for enabled in combinations(proposals, size):
            key = campaign.base_key
            if enabled:
                key += "__" + "__".join(enabled)
            candidates[key] = _campaign_columns(campaign, proposals, enabled)
    return candidates


def feature_campaign_manifest(name: str) -> dict[str, object]:
    campaign = feature_campaign(name)
    return {
        "name": campaign.name,
        "base_key": campaign.base_key,
        "base_features": tuple(
            FEATURE_NAMES[index] for index in campaign.base_columns
        ),
        "proposals": {
            proposal: tuple(FEATURE_NAMES[index] for index in columns)
            for proposal, columns in campaign.proposals
        },
        "launch_start": campaign.launch_start,
        "report_logical_key": campaign.report_logical_key,
        "artifact_collection": campaign.artifact_collection,
        "operation": campaign.operation,
        "trigger_kinds": dict(campaign.trigger_kinds),
        "burn_in_steps": campaign.burn_in_steps,
        "search_event_rate": campaign.search_event_rate,
    }


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
