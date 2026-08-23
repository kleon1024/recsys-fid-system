"""Atomic simulated-release state derived from sequential launch evidence."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping


PASS_DECISION = "pass_unified_lt_nonnegative"

RELEASE_RESOURCE_PATHS = {
    "feed-calibrated-v3-ab": (
        "reports/launches/2026-08-23-feed-calibrated-v3-1m-gpu.json"
    ),
    "feed-digital-twin-v2-ab": (
        "reports/launches/2026-08-23-feed-digital-twin-v2-1m-gpu.json"
    ),
    "feature-lr-sequential-ab": (
        "reports/launches/2026-08-23-feature-lr-sequential-1m-gpu.json"
    ),
    "feature-lr-hash-content-split-ab": (
        "reports/launches/2026-08-23-feature-lr-hash-split-1m-gpu.json"
    ),
    "feature-lr-local-ablation-ab": (
        "reports/launches/2026-08-23-feature-lr-local-ablation-1m-gpu.json"
    ),
    "feature-lr-intent-trigger-ablation-ab": (
        "reports/launches/2026-08-23-feature-lr-intent-trigger-1m-gpu.json"
    ),
    "feature-lr-v2": "artifacts/models/feature-lr-v2",
    "feature-lr-v3-hash-split": "artifacts/models/feature-lr-v3-hash-split",
    "feature-lr-v4-local-ablation": (
        "artifacts/models/feature-lr-v4-local-ablation"
    ),
    "feature-lr-v5-intent-trigger": "artifacts/models/feature-lr-v5-intent-trigger",
    "kuairand-standard-calibration": (
        "reports/calibration/2026-08-23-kuairand-standard-calibration.json"
    ),
}


def release_resource_path(root: Path, logical_key: str) -> Path:
    try:
        return root / RELEASE_RESOURCE_PATHS[logical_key]
    except KeyError as error:
        raise ValueError(f"unknown release resource: {logical_key}") from error


def initial_release_state(
    key: str,
    artifact: Mapping[str, object],
    artifact_collection: str | None = None,
) -> dict[str, object]:
    return {
        "active_key": key,
        "active_artifact": dict(artifact),
        "active_artifact_collection": artifact_collection,
        "rollback_key": None,
        "rollback_artifact": None,
        "rollback_artifact_collection": None,
        "promoted_by_launch": None,
    }


def release_state_from_manifest(
    release: Mapping[str, object], expected_active_key: str
) -> dict[str, object]:
    active_key = str(release["active_control_key"])
    if active_key != expected_active_key:
        raise ValueError(
            f"campaign base {expected_active_key} differs from active {active_key}"
        )
    return {
        "active_key": active_key,
        "active_artifact": dict(release["active_control_artifact"]),
        "active_artifact_collection": release.get(
            "active_artifact_collection", release.get("artifact_collection")
        ),
        "rollback_key": release["rollback_key"],
        "rollback_artifact": release["rollback_artifact"],
        "rollback_artifact_collection": release.get(
            "rollback_artifact_collection", release.get("artifact_collection")
        ),
        "promoted_by_launch": release["promoted_by_launch"],
    }


def apply_launch_decision(
    state: Mapping[str, object],
    candidate_key: str,
    candidate_artifact: Mapping[str, object],
    decision: str,
    launch_id: str,
    candidate_artifact_collection: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Promote only a passing candidate; every other decision is a no-op."""
    prior_key = str(state["active_key"])
    prior_artifact = dict(state["active_artifact"])
    promoted = decision == PASS_DECISION
    if promoted:
        next_state = {
            "active_key": candidate_key,
            "active_artifact": dict(candidate_artifact),
            "active_artifact_collection": candidate_artifact_collection,
            "rollback_key": prior_key,
            "rollback_artifact": prior_artifact,
            "rollback_artifact_collection": state[
                "active_artifact_collection"
            ],
            "promoted_by_launch": launch_id,
        }
    else:
        next_state = dict(state)
    promotion = {
        "prior_active_key": prior_key,
        "prior_active_artifact": prior_artifact,
        "candidate_key": candidate_key,
        "candidate_artifact": dict(candidate_artifact),
        "decision": decision,
        "promoted": promoted,
        "resulting_active_key": next_state["active_key"],
        "resulting_active_artifact": next_state["active_artifact"],
        "resulting_active_artifact_collection": next_state[
            "active_artifact_collection"
        ],
        "rollback_key": next_state["rollback_key"],
        "rollback_artifact": next_state["rollback_artifact"],
        "rollback_artifact_collection": next_state[
            "rollback_artifact_collection"
        ],
    }
    return next_state, promotion


def release_manifest(
    report: Mapping[str, object], report_sha256: str
) -> dict[str, object]:
    state = report["release_state"]
    if not isinstance(state, Mapping):
        raise ValueError("release report is missing release_state")
    return {
        "schema_version": "simulated-feed-release-v1",
        "environment": "synthetic_simulator",
        "active_control_key": state["active_key"],
        "active_control_artifact": state["active_artifact"],
        "active_artifact_collection": state["active_artifact_collection"],
        "rollback_key": state["rollback_key"],
        "rollback_artifact": state["rollback_artifact"],
        "rollback_artifact_collection": state["rollback_artifact_collection"],
        "promoted_by_launch": state["promoted_by_launch"],
        "source_report": {
            "logical_key": report["report_logical_key"],
            "sha256": report_sha256,
        },
        "production_readiness": report["production_readiness"],
        "evidence_boundary": (
            "This is simulator release state, not a production deployment."
        ),
    }


def write_release_manifest(
    report_path: Path, report: Mapping[str, object], output_path: Path
) -> dict[str, object]:
    report_hash = sha256(report_path.read_bytes()).hexdigest()
    manifest = release_manifest(report, report_hash)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2) + "\n"
    with NamedTemporaryFile(
        "w", dir=output_path.parent, delete=False, encoding="utf-8"
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)
    return manifest
