"""Composite simulator authority assembled from evidence-bound task kernels."""

from __future__ import annotations

import json
from pathlib import Path

from fid_lab.launches.release_resources import file_sha256

from .components import (
    build_feed_component,
    build_local_component,
    build_supply_component,
)


EVIDENCE_REPORTS = {
    "benchmark": "reports/world-model/v4/benchmark.json",
    "policy_adapter": "reports/world-model/v4/policy-adapter.json",
    "ope": "reports/world-model/v4/randomized-ope.json",
    "shadow_seed25": "reports/world-model/v4/stateful-shadow-world-seed25.json",
    "shadow_seed27": "reports/world-model/v4/stateful-shadow-world-seed27-adapted.json",
    "rejected_world_adapter": (
        "reports/world-model/v4/world-adapter-rejected-seed31.json"
    ),
    "core_challenger": "reports/world-model/v4/core-bridge-challenger.json",
    "feed_runtime_behavior": (
        "reports/launches/2026-08-24-feed-behavior-external-mixture-v4-100k.json"
    ),
    "feed_ecosystem_consumer": (
        "reports/launches/2026-08-24-feed-ecosystem-v4-consumer-100k.json"
    ),
    "feed_ecosystem_provider": (
        "reports/launches/2026-08-24-feed-ecosystem-v4-provider-100k.json"
    ),
    "local_v4_dataset": (
        "reports/datasets/2026-08-24-local-neural-v4-request-log-manifest.json"
    ),
    "local_v4_training": (
        "reports/training/2026-08-24-poi-distribution-v4-training.json"
    ),
    "local_v4_coarse": (
        "reports/launches/2026-08-24-poi-distribution-v4-coarse-1m.json"
    ),
    "local_v4_fine": (
        "reports/launches/2026-08-24-poi-distribution-v4-fine-mix-200k.json"
    ),
    "local_v4_end_to_end": (
        "reports/launches/2026-08-24-poi-distribution-v4-e2e-500k.json"
    ),
    "request_retrieval_training": (
        "reports/training/2026-08-24-shared-retrieval-v4-aligned-training.json"
    ),
    "request_retrieval_launch": (
        "reports/launches/2026-08-24-shared-retrieval-v4-aligned-paired-500k.json"
    ),
    "supply_v4_launch": (
        "reports/launches/2026-08-24-poi-posting-scaled-v4.json"
    ),
}


def _load_evidence(root: Path):
    reports = {}
    lineage = {}
    for key, relative in EVIDENCE_REPORTS.items():
        path = root / relative
        if not path.exists():
            raise ValueError(f"world authority evidence missing: {relative}")
        reports[key] = json.loads(path.read_text())
        lineage[key] = {"logical_key": key, "sha256": file_sha256(path)}
    return reports, lineage


def build_composite_world_review(root: Path) -> dict:
    reports, lineage = _load_evidence(root)
    feed = build_feed_component(reports)
    local = build_local_component(root, reports)
    supply = build_supply_component(root, reports)
    core = reports["core_challenger"]
    components = {
        "feed_behavior": feed,
        "local_response": local,
        "supply_response": supply,
        "retention_and_commercialization": {
            "status": "measurement_only",
            "external_validation": "missing",
        },
        "unified_neural_scm": {
            "status": core["evaluation"]["decision"],
            "gates": core["evaluation"]["gates"],
            "reason": (
                "External Feed and synthetic Local feature semantics cannot be "
                "merged into one causal artifact without task-specific evidence."
            ),
        },
    }
    return {
        "schema": "composite-recommendation-world-review-v1",
        "epoch": "v4",
        "decision": (
            "promote_feed_local_and_supply_kernels"
            if feed["status"] == "eligible_simulator_authority"
            and local["status"] == "eligible_simulator_authority"
            and supply["status"] == "eligible_simulator_authority"
            else "hold_v3_authority"
        ),
        "components": components,
        "lineage": lineage,
        "rollback_epoch": "v3",
        "evidence_boundary": (
            "This promotes an external-data-calibrated Feed kernel and a "
            "causally tested synthetic neural Local kernel inside the simulator. "
            "It is not a production deployment, live A/B, or production LT estimate."
        ),
    }


def build_world_release(review_path: Path) -> dict:
    review = json.loads(review_path.read_text())
    if review.get("decision") != "promote_feed_local_and_supply_kernels":
        raise ValueError("simulator world release requires accepted task kernels")
    feed = review["components"]["feed_behavior"]
    if feed.get("status") != "eligible_simulator_authority":
        raise ValueError("Feed behavior component is not eligible")
    local = review["components"]["local_response"]
    if local.get("status") != "eligible_simulator_authority":
        raise ValueError("Local response component is not eligible")
    supply = review["components"]["supply_response"]
    if supply.get("status") != "eligible_simulator_authority":
        raise ValueError("Supply response component is not eligible")
    return {
        "schema": "composite-simulator-world-authority-v1",
        "epoch": review["epoch"],
        "active_components": {
            "feed_behavior": {
                "authority": "external_sequence_mixture_v4",
                "policy_artifact_sha256": feed["policy_artifact_sha256"],
                "response_world_artifact_sha256": feed[
                    "response_world_artifact_sha256"
                ],
                "catalog_sha256": feed["catalog_sha256"],
                "profile_sha256": feed["profile_sha256"],
            },
            "local_response": {
                "authority": "synthetic_neural_v4",
                "artifact_sha256": local["artifact_sha256"],
                "retrieval_artifact_sha256": local["retrieval_artifact_sha256"],
            },
            "supply_response": {
                "authority": "synthetic_neural_v4",
                "artifact_sha256": supply["artifact_sha256"],
            },
            "retention_and_commercialization": {"authority": "measurement_only"},
        },
        "rollback_epoch": review["rollback_epoch"],
        "source_review_sha256": file_sha256(review_path),
        "production_readiness": "simulator_only",
        "evidence_boundary": review["evidence_boundary"],
    }
