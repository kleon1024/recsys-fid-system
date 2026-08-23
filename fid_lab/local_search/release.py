"""Hash-bound simulated authority for accepted Local Search ranking."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SOURCE_FILES = (
    "fid_lab/local_search/contracts.py",
    "fid_lab/local_search/simulation/world.py",
    "fid_lab/local_search/simulation/retrieval.py",
    "fid_lab/local_search/simulation/features.py",
    "fid_lab/local_search/simulation/samples.py",
    "fid_lab/local_search/simulation/response.py",
    "fid_lab/local_search/models/retrieval.py",
    "fid_lab/local_search/models/architectures.py",
    "fid_lab/local_search/models/ranking.py",
    "fid_lab/local_search/launch.py",
)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _resource(root, relative):
    return {"path": relative, "sha256": _hash(root / relative)}


def build_local_search_release(root, report_relative, artifact_relative):
    report = json.loads((root / report_relative).read_text())
    if report.get("schema") != "local-search-request-launch-review-v1":
        raise ValueError("Local Search release requires repeated launch review")
    state = report["release_state"]
    end = next(row for row in report["launches"] if row["stage"] == "end_to_end")
    if end["decision"] != "pass_all_seeds" or state["fine"] == "rule":
        raise ValueError("Local Search end-to-end proposal did not pass all seeds")
    seed_report = report["seed_reports"][0]
    artifact = seed_report["models"]["rankers"][state["fine"]]["artifact"]
    artifact_path = root / artifact_relative / artifact["artifact_file"]
    if _hash(artifact_path) != artifact["sha256"]:
        raise ValueError("Local Search artifact hash mismatch")
    active = {
        "retrieval_policy": state["retrieval"],
        "fine_model": state["fine"],
        "model_artifact": _resource(
            root, f"{artifact_relative}/{artifact['artifact_file']}"
        ),
        "model_seed": report["seeds"][0],
        "world_version": "teacher-hidden-local-search-v1",
        "sources": [_resource(root, relative) for relative in SOURCE_FILES],
    }
    encoded = json.dumps(active, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "simulated-local-search-authority-v1",
        "active_key": state["end_to_end"],
        "active_bundle_id": f"sha256:{sha256(encoded.encode()).hexdigest()}",
        "active_bundle": active,
        "rollback_key": "lexical_geo_plus_rule",
        "source_report": _resource(root, report_relative),
        "production_readiness": "hold_external_query_and_transaction_validation",
        "evidence_boundary": report["evidence_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    release = build_local_search_release(root, args.report, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({
        "active_key": release["active_key"],
        "production_readiness": release["production_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
