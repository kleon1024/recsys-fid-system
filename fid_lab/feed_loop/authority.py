"""One hash-bound authority for the active V3 Feed serving bundle."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ..simulation.environment import FEATURE_NAMES
from .models.artifact import feature_schema_hash


V3_SIGNAL = "kuairand-calibrated-v3"
V3_INDEX = "multiroute-rrf-coarse-v2"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _component(root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    return {"path": relative, "sha256": _hash(path)}


def _combined_hash(components: list[dict[str, str]]) -> str:
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


def rule_model_component(root: Path) -> dict[str, object]:
    sources = [
        _component(root, "fid_lab/feed_loop/tensor_policies.py"),
        _component(root, "fid_lab/feed_loop/tensor_cascade.py"),
    ]
    return {
        "kind": "parameterized_rule",
        "name": "personalized_rank",
        "artifact_id": f"sha256:{_combined_hash(sources)}",
        "sources": sources,
    }


def base_v3_components(root: Path) -> dict[str, object]:
    index_sources = [
        _component(root, "fid_lab/feed_loop/scale/graph/candidate.py"),
        _component(root, "fid_lab/feed_loop/scale/tensor_catalog.py"),
    ]
    behavior_sources = [
        _component(root, "fid_lab/feed_loop/scale/tensor_engine.py"),
        _component(root, "fid_lab/feed_loop/scale/calibration/behavior.py"),
        _component(root, "fid_lab/value/contracts.py"),
        _component(
            root,
            "reports/calibration/2026-08-23-kuairand-standard-calibration.json",
        ),
    ]
    feature_source = _component(
        root, "fid_lab/feed_loop/scale/artifact/features.py"
    )
    return {
        "index": {
            "version": V3_INDEX,
            "artifact_id": f"sha256:{_combined_hash(index_sources)}",
            "sources": index_sources,
            "catalog_items": 200_000,
        },
        "feature": {
            "schema_sha256": feature_schema_hash(),
            "artifact_id": f"sha256:{feature_source['sha256']}",
            "source": feature_source,
            "dense_fields": len(FEATURE_NAMES),
        },
        "behavior": {
            "signal_version": V3_SIGNAL,
            "artifact_id": f"sha256:{_combined_hash(behavior_sources)}",
            "sources": behavior_sources,
        },
    }


def build_rule_v3_authority(root: Path) -> dict[str, object]:
    historical = _component(
        root, "artifacts/releases/historical/simulated-feed-control-v2.json"
    )
    source_report = _component(
        root, "reports/launches/2026-08-23-feed-calibrated-v3-1m-gpu.json"
    )
    components = {"model": rule_model_component(root), **base_v3_components(root)}
    return {
        "schema_version": "simulated-feed-authority-v3",
        "environment": "externally_calibrated_synthetic_simulator",
        "epoch": "v3",
        "active_control_key": "rule_personalized_v1",
        "active_bundle_id": f"sha256:{_payload_hash(components)}",
        "active_bundle": components,
        "rollback_key": None,
        "rollback_bundle_id": None,
        "rollback_bundle": None,
        "dataset": None,
        "source_report": source_report,
        "historical_releases": [
            {
                "epoch": "v2",
                "status": "historical_not_serving_authority",
                **historical,
            }
        ],
        "production_readiness": "synthetic_research_only",
        "evidence_boundary": (
            "V3 is the current simulator authority; it is not a production deployment."
        ),
    }


def learned_model_component(root: Path, launch: dict[str, object],
                            artifact_dir: str) -> dict[str, object]:
    manifest = launch["artifact_manifest"]
    relative = f"{artifact_dir}/{manifest['artifact_file']}"
    artifact = _component(root, relative)
    if f"sha256:{artifact['sha256']}" != manifest["artifact_id"]:
        raise ValueError("launch artifact hash does not match the model file")
    sources = [
        _component(root, "fid_lab/feed_loop/models/feed_multitask.py"),
        _component(root, "fid_lab/value/predicted_tree.py"),
        _component(
            root, "fid_lab/feed_loop/scale/model_ladder/serving.py"
        ),
    ]
    return {
        "kind": "guarded_learned_residual_rerank",
        "name": launch["treatment"],
        "artifact_id": manifest["artifact_id"],
        "artifact": artifact,
        "model_manifest": manifest,
        "serving_policy": launch["serving_policy"],
        "sources": sources,
    }


def promote_v3_authority(root: Path, launch_report: str,
                         artifact_dir: str) -> dict[str, object]:
    current_path = root / "artifacts/releases/simulated-feed-control.json"
    current = json.loads(current_path.read_text())
    report_path = root / launch_report
    report = json.loads(report_path.read_text())
    active_key = report["release_state"]["active_key"]
    launch = next(
        row for row in report["launches"] if row["treatment"] == active_key
    )
    if not launch["promoted"] or not launch["decision"].startswith("pass_"):
        raise ValueError("the requested active launch did not pass promotion")
    rollback = {"model": rule_model_component(root), **base_v3_components(root)}
    if current["dataset"]["authority_bundle_id"] != (
        f"sha256:{_payload_hash(rollback)}"
    ):
        raise ValueError("training dataset is not bound to the rollback logger")
    active = {
        "model": learned_model_component(root, launch, artifact_dir),
        **base_v3_components(root),
    }
    current.update({
        "active_control_key": active_key,
        "active_bundle_id": f"sha256:{_payload_hash(active)}",
        "active_bundle": active,
        "rollback_key": "rule_personalized_v1",
        "rollback_bundle_id": f"sha256:{_payload_hash(rollback)}",
        "rollback_bundle": rollback,
        "source_report": _component(root, launch_report),
    })
    write_authority(root, current)
    return current


def write_authority(root: Path, authority: dict[str, object]) -> Path:
    path = root / "artifacts/releases/simulated-feed-control.json"
    path.write_text(json.dumps(authority, indent=2) + "\n")
    return path


def attach_dataset(
    root: Path, dataset_manifest: dict[str, object]
) -> dict[str, object]:
    path = root / "artifacts/releases/simulated-feed-control.json"
    authority = json.loads(path.read_text())
    if authority["epoch"] != "v3":
        raise ValueError("dataset can attach only to the active V3 authority")
    authority["dataset"] = dataset_manifest
    write_authority(root, authority)
    return authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("initialize-rule", "promote-model")
    )
    parser.add_argument("--launch-report")
    parser.add_argument(
        "--artifact-dir", default="artifacts/models/v3-model-ladder"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.mode == "initialize-rule":
        path = write_authority(root, build_rule_v3_authority(root))
    else:
        if not args.launch_report:
            parser.error("--launch-report is required for promote-model")
        promote_v3_authority(root, args.launch_report, args.artifact_dir)
        path = root / "artifacts/releases/simulated-feed-control.json"
    print(path)


if __name__ == "__main__":
    main()
