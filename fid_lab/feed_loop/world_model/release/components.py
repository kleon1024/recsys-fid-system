"""Evidence gates for independently owned simulator world components."""

from __future__ import annotations

from pathlib import Path

from fid_lab.launches.release_resources import file_sha256


def _all_gates(report: dict) -> bool:
    gates = report.get("gates", {})
    return bool(gates) and all(gates.values())


def _dataset_catalog_hash(report: dict) -> str:
    manifest = report.get("dataset_manifest") or report["artifacts"][
        "dataset_manifest"
    ]
    return manifest["catalog_sha256"]


def _launch_pass(report: dict, stage: str, treatment: str) -> bool:
    return any(
        row["stage"] == stage
        and row["treatment"] == treatment
        and row["decision"].startswith("pass")
        for row in report["launches"]
    )


def build_feed_component(reports: dict) -> dict:
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
    effects = {
        "randomized_dr_ope": ope["metrics"]["stay_norm"],
        "shadow_seed25": shadows[0]["metrics"]["stay_norm"],
        "shadow_seed27": shadows[1]["metrics"]["stay_norm"],
    }
    simulated_ab_pass = all(
        report["simulated_ab"]["decision"] == "simulated_ab_pass"
        and all(report["simulated_ab"]["gates"].values())
        for report in shadows
    )
    runtime = reports["feed_runtime_behavior"]
    consumer = reports["feed_ecosystem_consumer"]
    provider = reports["feed_ecosystem_provider"]
    runtime_world = runtime["behavior_world"]
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
        "runtime_behavior_review": runtime["decision"] == "feed_v4_behavior_pass"
        and _all_gates(runtime),
        "runtime_response_world_is_shadow_member": runtime_world[
            "artifact_sha256"
        ] in world_hashes,
        "runtime_catalog_matches_randomized_data": runtime_world[
            "catalog_sha256"
        ] == _dataset_catalog_hash(ope),
        "consumer_ecosystem_launch": consumer["decision"] == "ecosystem_v4_pass"
        and _all_gates(consumer),
        "creator_retention_launch": provider["decision"] == "ecosystem_v4_pass"
        and _all_gates(provider),
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
        "policy_artifact_sha256": next(iter(treatment_hashes)),
        "response_world_artifact_sha256": runtime_world["artifact_sha256"],
        "response_world_challenger_sha256": next(
            value for value in world_hashes
            if value != runtime_world["artifact_sha256"]
        ),
        "catalog_sha256": runtime_world["catalog_sha256"],
        "profile_sha256": runtime_world["profile_sha256"],
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


def build_local_component(root: Path, reports: dict) -> dict:
    dataset = reports["local_v4_dataset"]
    training = reports["local_v4_training"]
    artifact = training["models"]["linear"]["artifact"]
    artifact_path = root / "artifacts/models/poi-distribution-v4" / artifact[
        "artifact_file"
    ]
    retrieval = reports["request_retrieval_training"]["models"]["two_tower"][
        "artifact"
    ]
    retrieval_path = root / "artifacts/models/shared-retrieval-v4-aligned" / (
        retrieval["artifact_file"]
    )
    gates = {
        "hidden_neural_dgp": dataset["config"]["signal_version"]
        == "kuairand-local-neural-v4",
        "propensity_request_log": dataset["config"]["epsilon"] > 0,
        "model_artifact_bound": artifact_path.exists()
        and file_sha256(artifact_path) == artifact["sha256"],
        "coarse_launch": _launch_pass(
            reports["local_v4_coarse"], "coarse", "poi_coarse_linear"
        ),
        "fine_launch": _launch_pass(
            reports["local_v4_fine"], "fine", "poi_fine_linear"
        ),
        "end_to_end_launch": _launch_pass(
            reports["local_v4_end_to_end"], "end_to_end",
            "poi_e2e_linear_coarse_fine",
        ),
        "retrieval_artifact_bound": retrieval_path.exists()
        and file_sha256(retrieval_path) == retrieval["sha256"],
        "retrieval_launch": _launch_pass(
            reports["request_retrieval_launch"], "retrieval", "poi_ann_two_tower"
        ),
    }
    return {
        "scope": "local_response_and_poi_distribution",
        "status": "eligible_simulator_authority" if all(gates.values())
        else "hold_research_challenger",
        "world_version": "kuairand-local-neural-v4",
        "artifact_sha256": artifact["sha256"],
        "retrieval_artifact_sha256": retrieval["sha256"],
        "gates": gates,
        "external_validation": "missing",
        "not_authorized": (
            "production_local_lift", "real_transaction_lift",
            "production_exchange_rate",
        ),
    }


def build_supply_component(root: Path, reports: dict) -> dict:
    report = reports["supply_v4_launch"]
    state = report["release_state"]
    artifact = report["models_by_seed"][0][state["fine"]]["artifact"]
    artifact_path = root / "artifacts/models/poi-posting-v4" / artifact[
        "artifact_file"
    ]
    end = next(row for row in report["launches"] if row["stage"] == "end_to_end")
    gates = {
        "creator_panel_world": report["config"]["world_version"]
        == "creator-neural-supply-v4",
        "powered_creator_cluster_experiment": all(
            row["creator_online_ab"]["publish_rate"]["estimator"]
            == "cluster_randomized_ab_from_means"
            and row["requests"] >= 10_000_000
            for row in report["launches"]
            if row["stage"] in {"fine_scaled", "fine_scaled_incremental", "end_to_end"}
        ),
        "mature_relevance_mask": all(
            value["logging_contract"]["unmatured_relevance_uses_label_mask_zero"]
            for value in report["seed_diagnostics"]
        ),
        "model_artifact_bound": artifact_path.exists()
        and file_sha256(artifact_path) == artifact["artifact_sha256"],
        "end_to_end_launch": end["decision"].startswith("pass"),
        "complex_challengers_retained": all(
            any(
                row["stage"] == "fine_scaled_incremental"
                and row["treatment"] == treatment
                and not row["decision"].startswith("pass")
                for row in report["launches"]
            )
            for treatment in ("wide_deep", "mmoe")
        ),
    }
    return {
        "scope": "poi_posting_and_supply_to_feed",
        "status": "eligible_simulator_authority" if all(gates.values())
        else "hold_research_challenger",
        "world_version": report["config"]["world_version"],
        "artifact_sha256": artifact["artifact_sha256"],
        "gates": gates,
        "external_validation": "missing",
        "not_authorized": (
            "production_creator_lift", "production_supply_lift",
            "production_feed_distribution_lift",
        ),
    }
