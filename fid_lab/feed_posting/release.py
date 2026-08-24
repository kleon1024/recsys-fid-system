"""Hash-bound simulated authority for the accepted Feed-posting stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..launches.release_resources import (
    bundle_identifier, resource, source_resources, verified_artifact,
)
from .serving import policy_name

SHARED_SOURCES = (
    "fid_lab/launches/release_resources.py",
    "fid_lab/launches/statistics.py",
    "fid_lab/training/common/tensor_ops.py",
    "fid_lab/multitask.py",
)


def _ecosystem_evidence(root, relative, fine_model, blend, blend_mode):
    if relative is None:
        return None
    report = json.loads((root / relative).read_text())
    expected_treatment = policy_name(fine_model, blend, blend_mode)
    if (
        report.get("schema") != "feed-creator-ecosystem-v4-launch-review-v1"
        or report.get("decision") != "ecosystem_v4_pass"
        or report.get("ecosystem_config", {}).get("objective")
        != "posting_mediation"
        or report.get("control", {}).get("posting_policy")
        != policy_name(fine_model, 0.0, blend_mode)
        or report.get("treatment", {}).get("posting_policy")
        != expected_treatment
        or not all(report.get("gates", {}).values())
    ):
        raise ValueError("Feed-posting ecosystem mediation did not pass")
    return resource(root, relative)


def build_feed_posting_release(
    root, report_relative, artifact_relative, powered_ab_relative=None,
    ecosystem_relative=None, previous_release_relative=None,
):
    report = json.loads((root / report_relative).read_text())
    if report.get("schema") not in {
        "feed-posting-request-launch-review-v1",
        "feed-posting-request-launch-review-v2",
    }:
        raise ValueError("Feed-posting release requires repeated launch review")
    state = report["release_state"]
    powered = (
        None if powered_ab_relative is None
        else json.loads((root / powered_ab_relative).read_text())
    )
    if powered is None:
        end = next(
            row for row in report["launches"] if row["stage"] == "end_to_end"
        )
        if not end["decision"].startswith("pass") or state["fine"] == "rule":
            raise ValueError(
                "Feed-posting end-to-end proposal did not pass all seeds"
            )
        fine_model = state["fine"]
        active_key = state["end_to_end"]
    else:
        if (
            powered.get("schema") not in {
                "partitioned-feed-posting-v4-ab-v2",
                "partitioned-feed-posting-v4-ab-v3",
            }
            or powered.get("decision") != "pass"
            or powered.get("decision_estimator")
            != "creator_cluster_randomized_ab"
            or not all(powered.get("gates", {}).values())
        ):
            raise ValueError("Feed-posting powered creator A/B did not pass")
        fine_model = powered["treatment"].removeprefix(
            "trending_i2i_plus_"
        )
        active_key = (
            "trending_i2i_plus_"
            + policy_name(
                fine_model, powered["treatment_blend"],
                powered.get("treatment_blend_mode", "legacy_convex"),
            )
        )
    models = (
        report["models_by_seed"][0]
        if report["schema"].endswith("v2")
        else report["seed_reports"][0]["models"]
    )
    artifact = models[fine_model]["artifact"]
    model_artifact = verified_artifact(root, artifact_relative, artifact)
    if powered is not None and model_artifact["sha256"] != powered["model_sha256"]:
        raise ValueError("Feed-posting powered A/B model hash mismatch")
    blend = 1.0 if powered is None else powered["treatment_blend"]
    blend_mode = (
        "legacy_convex" if powered is None
        else powered.get("treatment_blend_mode", "legacy_convex")
    )
    ecosystem_evidence = _ecosystem_evidence(
        root, ecosystem_relative, fine_model, blend, blend_mode
    )
    source_packages = (
        "fid_lab/feed_posting",
        *(("fid_lab/feed_loop/ecosystem",) if ecosystem_evidence else ()),
    )
    active = {
        "candidate_policy": state["candidate"],
        "fine_model": fine_model,
        "model_blend": blend,
        "model_blend_mode": blend_mode,
        "training_objective": models[fine_model].get(
            "objective", "masked_conditional_v1"
        ),
        "model_artifact": model_artifact,
        "model_seed": report["seeds"][0],
        "world_version": report["config"].get(
            "world_version", "teacher-hidden-feed-posting-v1"
        ),
        "sources": source_resources(root, source_packages, SHARED_SOURCES),
        "evidence_reports": (
            [] if ecosystem_evidence is None else [ecosystem_evidence]
        ),
    }
    previous = (
        None if previous_release_relative is None
        else json.loads((root / previous_release_relative).read_text())
    )
    if previous is not None and previous.get("schema") != (
        "simulated-feed-posting-authority-v2"
    ):
        raise ValueError("Feed-posting rollback authority is invalid")
    return {
        "schema": "simulated-feed-posting-authority-v2",
        "active_key": active_key,
        "active_bundle_id": bundle_identifier(active),
        "active_bundle": active,
        "rollback_key": (
            "trending_i2i_plus_rule" if previous is None
            else previous["active_key"]
        ),
        "rollback_bundle_id": (
            None if previous is None else previous["active_bundle_id"]
        ),
        "previous_release": (
            None if previous_release_relative is None
            else resource(root, previous_release_relative)
        ),
        "source_report": resource(root, report_relative),
        "powered_ab_report": (
            None if powered_ab_relative is None
            else resource(root, powered_ab_relative)
        ),
        "ecosystem_report": ecosystem_evidence,
        "production_readiness": "hold_external_creator_and_supply_validation",
        "evidence_boundary": report["evidence_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--powered-ab")
    parser.add_argument("--ecosystem-report")
    parser.add_argument("--previous-release")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    release = build_feed_posting_release(
        root, args.report, args.artifact_dir, args.powered_ab,
        args.ecosystem_report, args.previous_release,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({
        "active_key": release["active_key"],
        "production_readiness": release["production_readiness"],
    }, indent=2))


if __name__ == "__main__":
    main()
