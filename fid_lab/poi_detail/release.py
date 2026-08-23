"""Hash-bound simulated authority for the POI Detail page control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..launches.release_resources import (
    bundle_identifier,
    resource,
    source_resources,
    verified_artifact,
)


SHARED_SOURCES = (
    "fid_lab/launches/statistics.py",
    "fid_lab/training/common/tensor_ops.py",
    "fid_lab/multitask.py",
    "fid_lab/value/contracts.py",
)


def _active_artifact(root, report, artifact_relative):
    fine = report["release_state"]["fine"]
    if fine == "rule":
        return resource(root, "fid_lab/poi_detail/simulation/features.py")
    evidence = report["seed_reports"][0]["models"][fine]["artifact"]
    return verified_artifact(root, artifact_relative, evidence)


def build_poi_detail_release(root, report_relative, artifact_relative):
    report = json.loads((root / report_relative).read_text())
    if report.get("schema") != "poi-detail-request-launch-review-v1":
        raise ValueError("POI Detail release requires repeated launch review")
    state = report["release_state"]
    active = {
        "mix_policy": "fixed_quota_4_2_2",
        "fine_model": state["fine"],
        "model_artifact": _active_artifact(root, report, artifact_relative),
        "model_seed": report["seeds"][0],
        "world_version": "teacher-hidden-poi-detail-v1",
        "sources": source_resources(
            root, "fid_lab/poi_detail", SHARED_SOURCES
        ),
    }
    return {
        "schema": "simulated-poi-detail-authority-v1",
        "active_key": state["end_to_end"],
        "active_bundle_id": bundle_identifier(active),
        "active_bundle": active,
        "rollback_key": "quota_mix_plus_rule",
        "source_report": resource(root, report_relative),
        "production_readiness": (
            "hold_external_page_transaction_and_review_validation"
        ),
        "evidence_boundary": report["evidence_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    release = build_poi_detail_release(
        root, args.report, args.artifact_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({
        "active_key": release["active_key"],
        "production_readiness": release["production_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
