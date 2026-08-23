"""Hash-bound simulated authority for accepted Local Search ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..launches.release_resources import (
    bundle_identifier, resource, source_resources, verified_artifact,
)

SHARED_SOURCES = (
    "fid_lab/launches/release_resources.py",
    "fid_lab/launches/statistics.py",
    "fid_lab/training/common/tensor_ops.py",
    "fid_lab/multitask.py",
)


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
    active = {
        "retrieval_policy": state["retrieval"],
        "fine_model": state["fine"],
        "model_artifact": verified_artifact(root, artifact_relative, artifact),
        "model_seed": report["seeds"][0],
        "world_version": "teacher-hidden-local-search-v1",
        "sources": source_resources(
            root, "fid_lab/local_search", SHARED_SOURCES
        ),
    }
    return {
        "schema": "simulated-local-search-authority-v1",
        "active_key": state["end_to_end"],
        "active_bundle_id": bundle_identifier(active),
        "active_bundle": active,
        "rollback_key": "lexical_geo_plus_rule",
        "source_report": resource(root, report_relative),
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
