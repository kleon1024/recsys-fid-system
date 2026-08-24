"""Combine training evidence with powered Supply V4 creator A/B reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ....launches.release_resources import resource


def _load(root, relative):
    return json.loads((root / relative).read_text())


def _scaled_row(report, stage, decision=None):
    return {
        "stage": stage,
        "control": report["control"].removeprefix("popular_geo_plus_"),
        "treatment": report["treatment"].removeprefix("popular_geo_plus_"),
        "metrics": report["metrics"],
        "gates": report["gates"],
        "decision": decision or (
            "pass_powered_creator_ab"
            if report["decision"] == "pass" else "hold_or_reject_powered_creator_ab"
        ),
        "creator_online_ab": report["creator_randomized_ab"],
        "requests": report["requests"],
        "creators": report["creators"],
        "model_sha256": report["model_sha256"],
        "control_model_sha256": report.get("control_model_sha256"),
    }


def build_scaled_posting_review(
    root: Path, training_relative: str, linear_relative: str,
    wide_relative: str, mmoe_relative: str,
):
    training = _load(root, training_relative)
    linear = _load(root, linear_relative)
    wide = _load(root, wide_relative)
    mmoe = _load(root, mmoe_relative)
    if training.get("schema") != "poi-posting-request-launch-review-v3":
        raise ValueError("scaled review requires Supply V4 training evidence")
    artifacts = training["models_by_seed"][0]
    expected = {
        name: artifacts[name]["artifact"]["artifact_sha256"]
        for name in ("linear", "wide_deep", "mmoe")
    }
    if linear["model_sha256"] != expected["linear"]:
        raise ValueError("Linear scaled A/B artifact does not match training")
    if wide["model_sha256"] != expected["wide_deep"]:
        raise ValueError("W&D scaled A/B artifact does not match training")
    if mmoe["model_sha256"] != expected["mmoe"]:
        raise ValueError("MMoE scaled A/B artifact does not match training")
    if any(
        report["control_model_sha256"] != expected["linear"]
        for report in (wide, mmoe)
    ):
        raise ValueError("complex challenger did not use Linear as control")
    if linear["decision"] != "pass":
        raise ValueError("Linear did not pass the powered creator A/B")
    if any(report["decision"] == "pass" for report in (wide, mmoe)):
        raise ValueError("complex challenger unexpectedly passed; review selection")
    candidates = [
        row for row in training["launches"] if row["stage"] == "candidate"
    ]
    linear_row = _scaled_row(linear, "fine_scaled")
    end_row = dict(linear_row)
    end_row.update({
        "stage": "end_to_end",
        "control": "popular_geo_plus_rule",
        "treatment": "popular_geo_plus_linear",
    })
    return {
        "schema": "poi-posting-scaled-launch-review-v4",
        "config": training["config"],
        "seeds": training["seeds"],
        "models_by_seed": training["models_by_seed"],
        "seed_diagnostics": training["seed_diagnostics"],
        "launches": [
            *candidates,
            linear_row,
            _scaled_row(wide, "fine_scaled_incremental", "reject_scaled_incremental"),
            _scaled_row(mmoe, "fine_scaled_incremental", "reject_scaled_incremental"),
            end_row,
        ],
        "release_state": {
            "candidate": "popular_geo",
            "fine": "linear",
            "end_to_end": "popular_geo_plus_linear",
        },
        "evidence": {
            "training": resource(root, training_relative),
            "linear_powered_ab": resource(root, linear_relative),
            "wide_deep_incremental_ab": resource(root, wide_relative),
            "mmoe_incremental_ab": resource(root, mmoe_relative),
        },
        "evidence_boundary": (
            "Training and offline metrics use 400K synthetic requests. Promotion "
            "uses a separate 10M-request, 1.25M-creator paired and randomized A/B. "
            "This is simulator launch evidence, not production lift."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--linear-ab", required=True)
    parser.add_argument("--wide-ab", required=True)
    parser.add_argument("--mmoe-ab", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    report = build_scaled_posting_review(
        root, args.training, args.linear_ab, args.wide_ab, args.mmoe_ab
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["release_state"], indent=2))


if __name__ == "__main__":
    main()
