"""Atomic simulated-release state derived from sequential launch evidence."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping


PASS_DECISION = "pass_unified_lt_nonnegative"


def initial_release_state(
    key: str, artifact: Mapping[str, object]
) -> dict[str, object]:
    return {
        "active_key": key,
        "active_artifact": dict(artifact),
        "rollback_key": None,
        "rollback_artifact": None,
        "promoted_by_launch": None,
    }


def apply_launch_decision(
    state: Mapping[str, object],
    candidate_key: str,
    candidate_artifact: Mapping[str, object],
    decision: str,
    launch_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Promote only a passing candidate; every other decision is a no-op."""
    prior_key = str(state["active_key"])
    prior_artifact = dict(state["active_artifact"])
    promoted = decision == PASS_DECISION
    if promoted:
        next_state = {
            "active_key": candidate_key,
            "active_artifact": dict(candidate_artifact),
            "rollback_key": prior_key,
            "rollback_artifact": prior_artifact,
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
        "rollback_key": next_state["rollback_key"],
        "rollback_artifact": next_state["rollback_artifact"],
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
        "rollback_key": state["rollback_key"],
        "rollback_artifact": state["rollback_artifact"],
        "promoted_by_launch": state["promoted_by_launch"],
        "source_report": {
            "logical_key": "feature-lr-sequential-ab",
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
