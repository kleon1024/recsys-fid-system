"""Hash-bound simulated authority for the accepted POI posting stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...launches.release_resources import (
    bundle_identifier, resource, source_resources, verified_artifact,
)

SHARED_SOURCES = (
    "fid_lab/launches/statistics.py",
    "fid_lab/training/common/tensor_ops.py",
    "fid_lab/multitask.py",
)


def build_posting_release(root, report_relative, artifact_relative):
    report_path = root / report_relative
    report = json.loads(report_path.read_text())
    if report.get("schema") != "poi-posting-request-launch-review-v2":
        raise ValueError("POI posting release requires the repeated launch review")
    state = report["release_state"]
    end = next(row for row in report["launches"] if row["stage"] == "end_to_end")
    if end["decision"] != "pass_all_seeds" or state["fine"] == "rule":
        raise ValueError("POI posting end-to-end proposal did not pass all seeds")
    seed_report = report["seed_reports"][0]
    artifact_manifest = seed_report["models"][state["fine"]]["artifact"]
    active = {
        "candidate_policy": state["candidate"],
        "fine_model": state["fine"],
        "model_artifact": verified_artifact(
            root, artifact_relative, artifact_manifest, "artifact_sha256"
        ),
        "model_seed": report["seeds"][0],
        "world_version": "teacher-hidden-posting-v1",
        "sources": source_resources(
            root, "fid_lab/poi_posting/world", SHARED_SOURCES
        ),
    }
    return {
        "schema": "simulated-poi-posting-authority-v1",
        "active_key": state["end_to_end"],
        "active_bundle_id": bundle_identifier(active),
        "active_bundle": active,
        "rollback_key": "popular_geo_plus_rule",
        "source_report": resource(root, report_relative),
        "production_readiness": "hold_external_creator_and_supply_validation",
        "evidence_boundary": report["evidence_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    release = build_posting_release(root, args.report, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({
        "active_key": release["active_key"],
        "production_readiness": release["production_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
