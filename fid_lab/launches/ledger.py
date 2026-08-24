"""Validated, queryable Launch Review ledger across recommendation surfaces."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .feed_posting_v43 import build_feed_posting_v43_records


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
        "pass_composite_simulator_kernels"
        if report["decision"] in {
            "promote_feed_kernel_only",
            "promote_feed_local_and_supply_kernels",
        } else "hold_world_authority"
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


def _retrieval_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-24-feed-retrieval-launch-review.json"
    report, evidence = _load(root, relative)
    return [
        _record(
            launch_id=f"L-RECALL-EXT-{index:03d}",
            surface="main_feed",
            stage="retrieval",
            change_type="retrieval_model",
            control=row["control"],
            treatment=row["treatment"],
            decision=row["decision"],
            evidence=evidence,
            primary_metric="random_exposure_recall_then_fixed_rank_stay",
            evidence_boundary=report["evidence_boundary"],
        )
        for index, row in enumerate(report["launches"], 1)
    ]


def _feed_v4_request_model_record(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-24-feed-v4-mmoe-guarded-010-1m.json"
    report, evidence = _load(root, relative)
    treatment = report["treatment"]
    return [_record(
        launch_id="L-FEED-V4-FINE-001",
        surface="main_feed",
        stage="fine",
        change_type="request_sequence_multitask_model",
        control=report["control"],
        treatment=treatment["name"],
        decision=(
            "pass_simulator_launch"
            if report["decision"] == "launch" else "hold_simulator_launch"
        ),
        evidence=evidence,
        primary_metric="unified_lt_with_negative_and_duration_guardrails",
        evidence_boundary=report["evidence_boundary"],
    )]


def _poi_distribution_stage_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-24-poi-distribution-stage-ladder.json"
    report, evidence = _load(root, relative)
    stage_counter = {stage: 0 for stage in ("retrieval", "coarse", "fine", "mix")}
    records = []
    for row in report["launches"]:
        stage = row["stage"]
        stage_counter[stage] += 1
        records.append(_record(
            launch_id=(
                f"L-POI-{stage.upper()}-{stage_counter[stage]:03d}"
            ),
            surface="poi_distribution",
            stage=stage,
            change_type=f"{stage}_evolution",
            control=row["control"],
            treatment=row["treatment"],
            decision=row["decision"],
            evidence=evidence,
            primary_metric=row["primary_metric"],
            evidence_boundary=report["evidence_boundary"],
        ))
    return records


def _surface_stage_records(
    root: Path, *, relative: str, surface: str, launch_prefix: str,
    stages: tuple[str, ...], change_prefix: str, primary_metric: str,
) -> list[dict]:
    report, evidence = _load(root, relative)
    counters = {stage: 0 for stage in stages}
    records = []
    for row in report["launches"]:
        stage = row["stage"]
        counters[stage] += 1
        records.append(_record(
            launch_id=f"{launch_prefix}-{stage.upper()}-{counters[stage]:03d}",
            surface=surface,
            stage=stage,
            change_type=f"{change_prefix}_{stage}_evolution",
            control=row["control"],
            treatment=row["treatment"],
            decision=row["decision"],
            evidence=evidence,
            primary_metric=primary_metric,
            evidence_boundary=report["evidence_boundary"],
        ))
    return records


def _poi_posting_stage_records(root: Path) -> list[dict]:
    return _surface_stage_records(
        root,
        relative="reports/launches/2026-08-24-poi-posting-request-launch-review.json",
        surface="poi_posting", launch_prefix="L-POI-POST",
        stages=("candidate", "fine", "end_to_end"), change_prefix="posting",
        primary_metric="publish_rate_with_platform_lt_and_content_risk",
    )


def _feed_posting_stage_records(root: Path) -> list[dict]:
    return _surface_stage_records(
        root,
        relative="reports/launches/2026-08-24-feed-posting-request-launch-review.json",
        surface="feed_posting", launch_prefix="L-FEED-POST",
        stages=("candidate", "fine", "end_to_end"),
        change_prefix="feed_posting",
        primary_metric="publish_rate_with_platform_lt_and_content_risk",
    )


def _feed_posting_v4_records(root: Path) -> list[dict]:
    ladder_relative = (
        "reports/launches/2026-08-24-feed-posting-v4-cascade-400k.json"
    )
    ladder, ladder_evidence = _load(root, ladder_relative)
    counters = {"candidate": 0, "fine": 0, "end_to_end": 0}
    records = []
    for row in ladder["launches"]:
        stage = "fine" if row["stage"] == "fine_incremental" else row["stage"]
        counters[stage] += 1
        records.append(_record(
            launch_id=f"L-FEED-POST-V4-{stage.upper()}-{counters[stage]:03d}",
            surface="feed_posting", stage=stage,
            change_type=f"cascade_v4_{row['stage']}",
            control=row["control"], treatment=row["treatment"],
            decision=row["decision"], evidence=ladder_evidence,
            primary_metric="creator_publish_with_platform_lt_and_content_risk",
            evidence_boundary=ladder["evidence_boundary"],
        ))
    powered_reports = (
        "reports/launches/2026-08-24-feed-posting-v4-din-005-ab-10m.json",
        "reports/launches/2026-08-24-feed-posting-v4-din-020-ab-10m.json",
    )
    for index, relative in enumerate(powered_reports, 1):
        report, evidence = _load(root, relative)
        records.append(_record(
            launch_id=f"L-FEED-POST-V4-POWERED-{index:03d}",
            surface="feed_posting", stage="end_to_end",
            change_type="creator_randomized_dose_ab",
            control=f"{report['control']}_blend_{report['control_blend']:.2f}",
            treatment=(
                f"{report['treatment']}_blend_{report['treatment_blend']:.2f}"
            ),
            decision=report["decision"], evidence=evidence,
            primary_metric="creator_randomized_publish_and_platform_lt",
            evidence_boundary=report["evidence_boundary"],
        ))
    mediation_relative = (
        "reports/launches/2026-08-24-feed-posting-cross-day-mediation-v4.json"
    )
    mediation, evidence = _load(root, mediation_relative)
    records.append(_record(
        launch_id="L-FEED-POST-V4-MEDIATION-001",
        surface="feed_posting", stage="end_to_end",
        change_type="cross_day_supply_to_feed_mediation",
        control=mediation["control"]["posting_policy"],
        treatment=mediation["treatment"]["posting_policy"],
        decision=(
            "pass_supply_primary_consumer_noninferior"
            if mediation["decision"] == "ecosystem_v4_pass"
            else "hold_cross_day_mediation"
        ),
        evidence=evidence,
        primary_metric="creator_posts_with_cross_day_feed_noninferiority",
        evidence_boundary=mediation["evidence_boundary"],
    ))
    return records


def _poi_posting_v4_records(root: Path) -> list[dict]:
    relative = "reports/launches/2026-08-24-poi-posting-scaled-v4.json"
    report, evidence = _load(root, relative)
    counters = {"candidate": 0, "fine": 0, "end_to_end": 0}
    records = []
    for row in report["launches"]:
        stage = (
            "fine" if row["stage"] in {
                "fine_incremental", "fine_scaled", "fine_scaled_incremental",
            } else row["stage"]
        )
        counters[stage] += 1
        records.append(_record(
            launch_id=f"L-POI-POST-V4-{stage.upper()}-{counters[stage]:03d}",
            surface="poi_posting", stage=stage,
            change_type=f"creator_neural_v4_{row['stage']}",
            control=row["control"], treatment=row["treatment"],
            decision=row["decision"], evidence=evidence,
            primary_metric="creator_publish_with_platform_lt_and_supply_quality",
            evidence_boundary=report["evidence_boundary"],
        ))
    return records


def _local_search_stage_records(root: Path) -> list[dict]:
    return _surface_stage_records(
        root,
        relative="reports/launches/2026-08-24-local-search-request-launch-review.json",
        surface="local_search", launch_prefix="L-LOCAL-SEARCH",
        stages=("retrieval", "fine", "end_to_end"),
        change_prefix="local_search",
        primary_metric="query_success_with_platform_lt_and_order_guardrails",
    )


def _poi_detail_stage_records(root: Path) -> list[dict]:
    return _surface_stage_records(
        root,
        relative="reports/launches/2026-08-24-poi-detail-request-launch-review.json",
        surface="poi_detail", launch_prefix="L-POI-DETAIL",
        stages=("fine", "end_to_end"), change_prefix="poi_detail",
        primary_metric="deep_action_with_platform_lt_and_safety_guardrails",
    )


def _poi_distribution_v4_records(root: Path) -> list[dict]:
    specifications = (
        (
            "reports/launches/2026-08-24-poi-distribution-v4-coarse-1m.json",
            {"coarse"},
        ),
        (
            "reports/launches/2026-08-24-poi-distribution-v4-fine-mix-200k.json",
            {"fine", "mix"},
        ),
        (
            "reports/launches/2026-08-24-poi-distribution-v4-e2e-500k.json",
            {"end_to_end"},
        ),
    )
    counters = {stage: 0 for stage in ("coarse", "fine", "mix", "end_to_end")}
    records = []
    for relative, included in specifications:
        report, evidence = _load(root, relative)
        for row in report["launches"]:
            stage = row["stage"]
            if stage not in included:
                continue
            counters[stage] += 1
            records.append(_record(
                launch_id=f"L-POI-V4-{stage.upper()}-{counters[stage]:03d}",
                surface="poi_distribution", stage=stage,
                change_type=f"trained_v4_{stage}", control=row["control"],
                treatment=row["treatment"], decision=row["decision"],
                evidence=evidence,
                primary_metric=(
                    "coarse_oracle_recall_with_platform_lt"
                    if stage == "coarse" else
                    "local_action_with_platform_lt_and_safety"
                ),
                evidence_boundary=report["evidence_boundary"],
            ))
    return records


def _request_retrieval_v4_records(root: Path) -> list[dict]:
    specifications = (
        (
            "reports/launches/2026-08-24-poi-retrieval-v4-poi-only-500k.json",
            "poi_distribution", "poi_only_corpus",
        ),
        (
            "reports/launches/2026-08-24-shared-retrieval-v4-sequence-skew-500k.json",
            "main_feed", "training_serving_skew",
        ),
        (
            "reports/launches/2026-08-24-shared-retrieval-v4-aligned-paired-500k.json",
            "main_feed", "aligned_query_paired_ab",
        ),
    )
    records = []
    for campaign, (relative, surface, change_type) in enumerate(specifications, 1):
        report, evidence = _load(root, relative)
        for model, row in enumerate(report["launches"], 1):
            records.append(_record(
                launch_id=f"L-REQUEST-RETRIEVAL-V4-{campaign:02d}-{model:02d}",
                surface=surface,
                stage="retrieval",
                change_type=change_type,
                control=row["control"],
                treatment=row["treatment"],
                decision=row["decision"],
                evidence=evidence,
                primary_metric="equal_corpus_recall_with_unified_lt_and_local_guardrails",
                evidence_boundary=report["evidence_boundary"],
            ))
    return records


def _unified_feed_local_serving_record(root: Path) -> list[dict]:
    relative = (
        "reports/launches/"
        "2026-08-24-unified-feed-local-serving-v1-1m.json"
    )
    report, evidence = _load(root, relative)
    return [_record(
        launch_id="L-SERVING-UNIFIED-001",
        surface="main_feed",
        stage="end_to_end",
        change_type="typed_feed_local_value_composition",
        control=report["control"]["name"],
        treatment=report["treatment"]["name"],
        decision=report["decision"],
        evidence=evidence,
        primary_metric="local_anchor_with_platform_lt_and_feed_guardrails",
        evidence_boundary=report["evidence_boundary"],
    )]


def build_launch_ledger(root: Path) -> dict:
    records = [
        *_main_feed_policy_records(root),
        *_fine_model_records(root),
        *_coarse_records(root),
        *_feature_records(root),
        *_local_records(root),
        *_world_records(root),
        *_retrieval_records(root),
        *_feed_v4_request_model_record(root),
        *_poi_distribution_stage_records(root),
        *_poi_posting_stage_records(root),
        *_feed_posting_stage_records(root),
        *_feed_posting_v4_records(root),
        *build_feed_posting_v43_records(root, _load, _record),
        *_local_search_stage_records(root),
        *_poi_detail_stage_records(root),
        *_poi_distribution_v4_records(root),
        *_request_retrieval_v4_records(root),
        *_poi_posting_v4_records(root),
        *_unified_feed_local_serving_record(root),
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
            "continued": sum(
                record["decision"].startswith("continue")
                for record in records
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
