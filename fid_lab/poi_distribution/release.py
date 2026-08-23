"""Hash-bound simulator authority for the accepted POI distribution V4 stack."""

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


def _passed(report, stage, treatment):
    row = next(
        value for value in report["launches"]
        if value["stage"] == stage and value["treatment"] == treatment
    )
    if not row["decision"].startswith("pass"):
        raise ValueError(f"POI distribution {stage} proposal did not pass")
    return row


def build_poi_distribution_release(
    root, training_relative, coarse_relative, fine_relative, end_relative,
    artifact_dir,
):
    training = json.loads((root / training_relative).read_text())
    coarse = json.loads((root / coarse_relative).read_text())
    fine = json.loads((root / fine_relative).read_text())
    end = json.loads((root / end_relative).read_text())
    if training.get("schema") != "poi-distribution-model-training-v1":
        raise ValueError("POI distribution release requires V4 training evidence")
    _passed(coarse, "coarse", "poi_coarse_linear")
    _passed(fine, "fine", "poi_fine_linear")
    _passed(end, "end_to_end", "poi_e2e_linear_coarse_fine")
    artifact = verified_artifact(
        root, artifact_dir, training["models"]["linear"]["artifact"]
    )
    evidence_reports = [
        resource(root, relative) for relative in (
            training_relative, coarse_relative, fine_relative, end_relative,
        )
    ]
    active = {
        "retrieval_policy": "ann_graph_geo_fresh_long_tail_popular_search_retarget",
        "coarse_model": "linear",
        "fine_model": "linear",
        "mix_policy": "feed_guarded_no_extra_local_weight",
        "model_artifact": artifact,
        "world_version": "kuairand-local-neural-v4",
        "training_dataset": resource(
            root,
            "reports/datasets/2026-08-24-local-neural-v4-request-log-manifest.json",
        ),
        "evidence_reports": evidence_reports,
        "sources": source_resources(root, (
            "fid_lab/poi_distribution",
            "fid_lab/feed_loop/scale/tensor_runtime/local_response.py",
            "fid_lab/feed_loop/scale/tensor_engine.py",
            "fid_lab/feed_loop/tensor_cascade.py",
            "fid_lab/simulation/experimentation/assignment.py",
            "fid_lab/launches/release_resources.py",
        )),
    }
    return {
        "schema": "simulated-poi-distribution-v4-authority-v1",
        "active_key": "eight_route_linear_coarse_linear_fine_feed_guarded_mix",
        "active_bundle_id": bundle_identifier(active),
        "active_bundle": active,
        "rollback_key": "eight_route_quality_coarse_rule_fine",
        "source_report": resource(root, end_relative),
        "production_readiness": "simulator_only_external_local_validation_required",
        "evidence_boundary": end["evidence_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--coarse", required=True)
    parser.add_argument("--fine", required=True)
    parser.add_argument("--end-to-end", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    release = build_poi_distribution_release(
        root, args.training, args.coarse, args.fine, args.end_to_end,
        args.artifact_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({
        "active_key": release["active_key"],
        "production_readiness": release["production_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
