"""Hash-bound simulated authority for the accepted POI posting stack."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SOURCE_FILES = (
    "fid_lab/poi_posting/world/contracts.py",
    "fid_lab/poi_posting/world/generator.py",
    "fid_lab/poi_posting/world/models.py",
    "fid_lab/poi_posting/world/launch.py",
)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _resource(root, relative):
    path = root / relative
    return {"path": relative, "sha256": _hash(path)}


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
    artifact_path = root / artifact_relative / artifact_manifest["artifact_file"]
    if _hash(artifact_path) != artifact_manifest["artifact_sha256"]:
        raise ValueError("POI posting artifact hash mismatch")
    active = {
        "candidate_policy": state["candidate"],
        "fine_model": state["fine"],
        "model_artifact": _resource(
            root, f"{artifact_relative}/{artifact_manifest['artifact_file']}"
        ),
        "model_seed": report["seeds"][0],
        "world_version": "teacher-hidden-posting-v1",
        "sources": [_resource(root, relative) for relative in SOURCE_FILES],
    }
    encoded = json.dumps(active, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "simulated-poi-posting-authority-v1",
        "active_key": state["end_to_end"],
        "active_bundle_id": f"sha256:{sha256(encoded.encode()).hexdigest()}",
        "active_bundle": active,
        "rollback_key": "popular_geo_plus_rule",
        "source_report": _resource(root, report_relative),
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
