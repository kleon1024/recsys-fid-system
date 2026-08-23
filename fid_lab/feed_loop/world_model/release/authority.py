"""Composite simulator authority assembled from evidence-bound task kernels."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


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
}


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_evidence(root: Path):
    reports = {}
    lineage = {}
    for key, relative in EVIDENCE_REPORTS.items():
        path = root / relative
        if not path.exists():
            raise ValueError(f"world authority evidence missing: {relative}")
        reports[key] = json.loads(path.read_text())
        lineage[key] = {"logical_key": key, "sha256": _hash(path)}
    return reports, lineage


def _all_gates(report):
    gates = report.get("gates", {})
    return bool(gates) and all(gates.values())


def _dataset_catalog_hash(report):
    manifest = report.get("dataset_manifest") or report["artifacts"][
        "dataset_manifest"
    ]
    return manifest["catalog_sha256"]


def _feed_component(reports):
    ope = reports["ope"]
    shadows = (reports["shadow_seed25"], reports["shadow_seed27"])
    catalog_hashes = {
        _dataset_catalog_hash(reports["benchmark"]),
        _dataset_catalog_hash(ope),
        *(_dataset_catalog_hash(report) for report in shadows),
    }
    treatment_hashes = {
        ope["artifacts"]["treatment"],
        *(report["artifacts"]["treatment"] for report in shadows),
    }
    world_hashes = {report["artifacts"]["world"] for report in shadows}
    simulated_ab_pass = all(
        report["simulated_ab"]["decision"] == "simulated_ab_pass"
        and all(report["simulated_ab"]["gates"].values())
        for report in shadows
    )
    effects = {
        "randomized_dr_ope": ope["metrics"]["stay_norm"],
        "shadow_seed25": shadows[0]["metrics"]["stay_norm"],
        "shadow_seed27": shadows[1]["metrics"]["stay_norm"],
    }
    gates = {
        "one_dataset_catalog": len(catalog_hashes) == 1,
        "one_treatment_artifact": len(treatment_hashes) == 1,
        "two_independent_shadow_worlds": len(world_hashes) == 2,
        "randomized_ope": ope.get("decision") == "randomized_ope_pass"
        and _all_gates(ope),
        "shadow_seed25": shadows[0].get("decision") == "stateful_shadow_pass"
        and _all_gates(shadows[0]),
        "shadow_seed27": shadows[1].get("decision") == "stateful_shadow_pass"
        and _all_gates(shadows[1]),
        "million_user_power_simulation": simulated_ab_pass,
        "failed_seed_retained": reports["rejected_world_adapter"].get("decision")
        == "adapter_reject",
        "primary_direction_agrees": all(
            metric["confidence_interval_95"][0] > 0
            for metric in effects.values()
        ),
    }
    ope_effect = effects["randomized_dr_ope"]["absolute_delta"]
    shadow_effects = [
        effects[name]["absolute_delta"]
        for name in ("shadow_seed25", "shadow_seed27")
    ]
    return {
        "scope": "feed_behavior_only",
        "status": "eligible_simulator_authority" if all(gates.values())
        else "hold_research_challenger",
        "dataset_catalog_sha256": next(iter(catalog_hashes)),
        "artifact_sha256": next(iter(treatment_hashes)),
        "independent_world_sha256": sorted(world_hashes),
        "gates": gates,
        "stay_norm_effects": effects,
        "shadow_to_ope_magnitude_ratio": [
            value / ope_effect for value in shadow_effects
        ],
        "labels": (
            "click", "long_view", "like", "comment", "forward", "follow",
            "hate", "stay_norm",
        ),
        "not_authorized": (
            "unified_lt", "retention", "poi", "supply", "transaction",
            "commercialization",
        ),
    }


def build_composite_world_review(root: Path) -> dict:
    reports, lineage = _load_evidence(root)
    feed = _feed_component(reports)
    core = reports["core_challenger"]
    components = {
        "feed_behavior": feed,
        "local_response": {
            "status": "synthetic_v3_authority",
            "external_validation": "missing",
        },
        "supply_response": {
            "status": "synthetic_v3_authority",
            "external_validation": "missing",
        },
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
            "promote_feed_kernel_only"
            if feed["status"] == "eligible_simulator_authority"
            else "hold_v3_authority"
        ),
        "components": components,
        "lineage": lineage,
        "rollback_epoch": "v3",
        "evidence_boundary": (
            "This promotes one external-data-calibrated Feed behavior kernel "
            "inside the simulator. It is not a production deployment, live A/B, "
            "unified LT estimate, or validation of Local and supply kernels."
        ),
    }


def build_world_release(review_path: Path) -> dict:
    review = json.loads(review_path.read_text())
    if review.get("decision") != "promote_feed_kernel_only":
        raise ValueError("simulator world release requires an accepted Feed kernel")
    feed = review["components"]["feed_behavior"]
    if feed.get("status") != "eligible_simulator_authority":
        raise ValueError("Feed behavior component is not eligible")
    return {
        "schema": "composite-simulator-world-authority-v1",
        "epoch": review["epoch"],
        "active_components": {
            "feed_behavior": {
                "authority": "external_randomized_v4",
                "artifact_sha256": feed["artifact_sha256"],
            },
            "local_response": {"authority": "synthetic_v3"},
            "supply_response": {"authority": "synthetic_v3"},
            "retention_and_commercialization": {"authority": "measurement_only"},
        },
        "rollback_epoch": review["rollback_epoch"],
        "source_review_sha256": _hash(review_path),
        "production_readiness": "simulator_only",
        "evidence_boundary": review["evidence_boundary"],
    }
