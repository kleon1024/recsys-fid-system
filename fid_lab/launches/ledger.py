"""Validated, queryable Launch Review ledger across recommendation surfaces."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


REQUIRED_CELLS = {
    "main_feed": ("retrieval", "coarse", "fine", "mix", "end_to_end"),
    "poi_distribution": ("retrieval", "coarse", "fine", "mix", "end_to_end"),
    "poi_posting": ("candidate", "fine", "end_to_end"),
    "feed_posting": ("candidate", "fine", "end_to_end"),
    "local_search": ("retrieval", "fine", "end_to_end"),
    "poi_detail": ("fine", "end_to_end"),
}


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root: Path, relative: str) -> tuple[dict, dict]:
    path = root / relative
    if not path.exists():
        raise ValueError(f"launch evidence does not exist: {relative}")
    return json.loads(path.read_text()), {
        "report": relative,
        "report_sha256": _hash(path),
    }


def _record(
    *, launch_id: str, surface: str, stage: str, change_type: str,
    control: str, treatment: str, decision: str, evidence: dict,
    primary_metric: str, evidence_boundary: str,
) -> dict:
    if not decision:
        raise ValueError(f"launch has no decision: {launch_id}")
    return {
        "launch_id": launch_id,
        "surface": surface,
        "stage": stage,
        "change_type": change_type,
        "control": control,
        "treatment": treatment,
        "primary_metric": primary_metric,
        "decision": decision,
        "evidence": evidence,
        "evidence_boundary": evidence_boundary,
    }


def _main_feed_policy_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-23-main-feed-launch-suite.json"
    report, evidence = _load(root, relative)
    stage = {
        "feature": "fine",
        "strategy": "mix",
        "realtime": "fine",
        "product": "end_to_end",
        "business_value": "mix",
        "long_term_value": "mix",
        "chain_diagnosis": "end_to_end",
    }
    return [
        _record(
            launch_id=row["spec"]["launch_id"],
            surface="main_feed",
            stage=stage[row["spec"]["category"]],
            change_type=row["spec"]["category"],
            control=row["spec"]["control"]["name"],
            treatment=row["spec"]["treatment"]["name"],
            decision=row["decision"],
            evidence=evidence,
            primary_metric=row["spec"]["primary_metric"],
            evidence_boundary="synthetic common-random user A/B",
        )
        for row in report["launches"]
    ]


def _fine_model_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-23-v3-model-ladder-1m-gpu.json"
    report, evidence = _load(root, relative)
    return [
        _record(
            launch_id=f"L-FINE-{index:03d}",
            surface="main_feed",
            stage="fine",
            change_type="model",
            control=row["control"],
            treatment=row["treatment"],
            decision=row["decision"],
            evidence=evidence,
            primary_metric="unified_lt",
            evidence_boundary=report["evidence_boundary"],
        )
        for index, row in enumerate(report["launches"], 1)
    ]


def _coarse_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-23-coarse-cascade-ladder.json"
    report, evidence = _load(root, relative)
    return [
        _record(
            launch_id=row["launch_id"],
            surface="main_feed",
            stage="coarse",
            change_type="model_or_budget",
            control=row["control"],
            treatment=next(
                launch["treatment"]
                for launch in report["replicates"][0]["launches"]
                if launch["launch_id"] == row["launch_id"]
            ),
            decision=row["decision"],
            evidence=evidence,
            primary_metric="coarse_oracle_recall_and_unified_lt",
            evidence_boundary=report["evidence_boundary"],
        )
        for row in report["aggregate"]
    ]


def _feature_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-23-feature-lr-sequential-1m-gpu.json"
    report, evidence = _load(root, relative)
    return [
        _record(
            launch_id=row["launch_id"],
            surface="main_feed",
            stage="fine",
            change_type="feature",
            control=row["control"],
            treatment=row["treatment"],
            decision=row["decision"],
            evidence=evidence,
            primary_metric="unified_lt",
            evidence_boundary=report["evidence_boundary"],
        )
        for row in report["launches"]
    ]


def _local_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-23-local-service-supply-launch.json"
    report, evidence = _load(root, relative)
    boundary = report["limitations"]
    return [
        _record(
            launch_id="L-POI-DIST-E2E-001",
            surface="poi_distribution",
            stage="end_to_end",
            change_type="model_and_value_tree",
            control="lr_plus_sequence",
            treatment="local_value_tree_policy",
            decision=report["launch_decision"],
            evidence=evidence,
            primary_metric="unified_lt_with_feed_guardrails",
            evidence_boundary=boundary,
        ),
        _record(
            launch_id="L-POI-POST-E2E-001",
            surface="poi_posting",
            stage="end_to_end",
            change_type="supply_strategy",
            control="control_posting",
            treatment="treatment_posting",
            decision=report["launch_decision"],
            evidence=evidence,
            primary_metric="published_supply_then_distribution_lt",
            evidence_boundary=boundary,
        ),
    ]


def _world_records(root: Path) -> list[dict]:
    relative = "reports/world-model/v4/composite-launch-review.json"
    report, evidence = _load(root, relative)
    feed = report["components"]["feed_behavior"]
    decision = (
        "pass_feed_kernel_only"
        if report["decision"] == "promote_feed_kernel_only"
        else "hold_world_authority"
    )
    return [_record(
        launch_id="L-SIMULATOR-009",
        surface="main_feed",
        stage="end_to_end",
        change_type="simulator_world",
        control="synthetic_v3_feed_behavior",
        treatment="external_randomized_v4_feed_behavior",
        decision=decision,
        evidence=evidence,
        primary_metric="randomized_stay_with_behavior_guardrails",
        evidence_boundary=(
            report["evidence_boundary"] + " Feed status: " + feed["status"]
        ),
    )]


def build_launch_ledger(root: Path) -> dict:
    records = [
        *_main_feed_policy_records(root),
        *_fine_model_records(root),
        *_coarse_records(root),
        *_feature_records(root),
        *_local_records(root),
        *_world_records(root),
    ]
    identifiers = [record["launch_id"] for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("launch IDs must be unique")
    covered = {(record["surface"], record["stage"]) for record in records}
    coverage = {
        surface: {
            stage: "evidenced" if (surface, stage) in covered else "missing"
            for stage in stages
        }
        for surface, stages in REQUIRED_CELLS.items()
    }
    return {
        "schema": "recommendation-launch-ledger-v1",
        "records": records,
        "coverage": coverage,
        "summary": {
            "records": len(records),
            "passed": sum(record["decision"].startswith("pass") for record in records),
            "held": sum(record["decision"].startswith("hold") for record in records),
            "rejected": sum(
                record["decision"].startswith("reject") for record in records
            ),
            "missing_cells": sum(
                value == "missing"
                for surface in coverage.values()
                for value in surface.values()
            ),
        },
        "evidence_boundary": (
            "The ledger indexes synthetic launch evidence. Missing cells remain "
            "missing and no record is company production evidence."
        ),
    }
