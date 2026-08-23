"""Hash-bound simulated authority for the accepted Feed-posting stack."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SOURCE_FILES = (
    "fid_lab/feed_posting/contracts.py",
    "fid_lab/feed_posting/simulation/world.py",
    "fid_lab/feed_posting/simulation/retrieval.py",
    "fid_lab/feed_posting/simulation/features.py",
    "fid_lab/feed_posting/simulation/response.py",
    "fid_lab/feed_posting/models.py",
    "fid_lab/feed_posting/launch.py",
)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _resource(root, relative):
    return {"path": relative, "sha256": _hash(root / relative)}


def build_feed_posting_release(root, report_relative, artifact_relative):
    report = json.loads((root / report_relative).read_text())
    if report.get("schema") != "feed-posting-request-launch-review-v1":
        raise ValueError("Feed-posting release requires repeated launch review")
    state = report["release_state"]
    end = next(row for row in report["launches"] if row["stage"] == "end_to_end")
    if end["decision"] != "pass_all_seeds" or state["fine"] == "rule":
        raise ValueError("Feed-posting end-to-end proposal did not pass all seeds")
    seed_report = report["seed_reports"][0]
    artifact = seed_report["models"][state["fine"]]["artifact"]
    artifact_path = root / artifact_relative / artifact["artifact_file"]
    if _hash(artifact_path) != artifact["sha256"]:
        raise ValueError("Feed-posting artifact hash mismatch")
    active = {
        "candidate_policy": state["candidate"],
        "fine_model": state["fine"],
        "model_artifact": _resource(
            root, f"{artifact_relative}/{artifact['artifact_file']}"
        ),
        "model_seed": report["seeds"][0],
        "world_version": "teacher-hidden-feed-posting-v1",
        "sources": [_resource(root, relative) for relative in SOURCE_FILES],
    }
    encoded = json.dumps(active, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "simulated-feed-posting-authority-v1",
        "active_key": state["end_to_end"],
        "active_bundle_id": f"sha256:{sha256(encoded.encode()).hexdigest()}",
        "active_bundle": active,
        "rollback_key": "trending_i2i_plus_rule",
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
    root = Path(__file__).resolve().parents[2]
    release = build_feed_posting_release(root, args.report, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({
        "active_key": release["active_key"],
        "production_readiness": release["production_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
