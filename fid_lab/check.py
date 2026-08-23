"""Single local acceptance gate for structure, tests, and system behavior."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

from .feed_loop.release import release_resource_path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (ROOT / "fid_lab", ROOT / "tests")
PUBLIC_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "REQUEST_FOR_PROPOSAL.md",
    "docs/procurement/bidder-response-template.md",
    "docs/architecture/visual-atlas.md",
    "SECURITY.md",
    "docs/architecture/poi-posting.md",
    "docs/architecture/model-suite.md",
    "docs/architecture/model-evolution.md",
    "docs/operations/failure-runbook.md",
    "docs/interview/project-deep-dive.md",
)


def python_files() -> list[Path]:
    return sorted(path for directory in SOURCE_DIRS for path in directory.rglob("*.py"))


def check_file(path: Path) -> list[str]:
    failures: list[str] = []
    text = path.read_text()
    lines = text.splitlines()
    if len(lines) > 800:
        failures.append(f"{path}: file exceeds 800 lines")
    failures.extend(
        f"{path}:{number}: trailing whitespace"
        for number, line in enumerate(lines, start=1)
        if line != line.rstrip()
    )
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > 120:
                failures.append(f"{path}:{node.lineno}: function exceeds 120 lines")
        if isinstance(node, ast.Global):
            failures.append(f"{path}:{node.lineno}: global reassignment is prohibited")
    return failures


def check_structure() -> None:
    failures = [failure for path in python_files() for failure in check_file(path)]
    if failures:
        raise SystemExit("\n".join(failures))


def check_public_docs() -> None:
    failures: list[str] = []
    texts: dict[str, str] = {}
    for name in PUBLIC_DOCS:
        path = ROOT / name
        if not path.exists():
            failures.append(f"missing public document: {name}")
            continue
        texts[name] = path.read_text()
    for path in sorted((ROOT / "docs").rglob("*.md")):
        texts[str(path.relative_to(ROOT))] = path.read_text()
    combined = "\n".join(texts.values())
    for banned in ("/Users/", "gho_", "BEGIN PRIVATE KEY", "TODO"):
        if banned in combined:
            failures.append(f"public documents contain prohibited text: {banned}")
    mermaid_starts = combined.count("```mermaid")
    mermaid_blocks = len(re.findall(r"```mermaid\n[\s\S]*?```", combined))
    if mermaid_starts != mermaid_blocks or mermaid_blocks < 10:
        failures.append("public diagrams are missing or have unbalanced Mermaid fences")
    rfp = texts.get("REQUEST_FOR_PROPOSAL.md", "")
    for heading in ("## 4. Scope of work", "## 7. Delivery gates and acceptance", "## 10. Evaluation rubric"):
        if heading not in rfp:
            failures.append(f"RFP missing required section: {heading}")
    if failures:
        raise SystemExit("\n".join(failures))


def check_report_manifests() -> None:
    for manifest_name in (
        "reports/launches/MANIFEST.sha256",
        "reports/benchmarks/MANIFEST.sha256",
    ):
        manifest = ROOT / manifest_name
        failures = []
        declared = set()
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            path = ROOT / relative
            declared.add(path.resolve())
            if not path.exists():
                failures.append(f"missing report: {relative}")
                continue
            actual = sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(f"report hash mismatch: {relative}")
        report_dir = manifest.parent
        present = {path.resolve() for path in report_dir.glob("*.json")}
        for path in sorted(present - declared):
            failures.append(f"report missing from manifest: {path.relative_to(ROOT)}")
        if failures:
            raise SystemExit("\n".join(failures))


def check_visual_manifest() -> None:
    manifest = ROOT / "docs/assets/MANIFEST.sha256"
    failures = []
    declared = set()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        declared.add(path.resolve())
        if not path.exists():
            failures.append(f"missing visual: {relative}")
            continue
        if sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"visual hash mismatch: {relative}")
    present = {path.resolve() for path in manifest.parent.glob("*.svg")}
    for path in sorted(present - declared):
        failures.append(f"visual missing from manifest: {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit("\n".join(failures))


def _model_manifest_failures(relative_manifest: str) -> list[str]:
    failures = []
    manifest = ROOT / relative_manifest
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing model artifact: {relative}")
        elif sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"model artifact hash mismatch: {relative}")
    return failures


def check_model_artifacts() -> None:
    manifests = (
        "artifacts/models/stateful-v2/MANIFEST.sha256",
        "artifacts/models/feature-lr-v2/MANIFEST.sha256",
        "artifacts/models/feature-lr-v3-hash-split/MANIFEST.sha256",
        "artifacts/models/feature-lr-v4-local-ablation/MANIFEST.sha256",
    )
    failures = []
    for relative_manifest in manifests:
        failures.extend(_model_manifest_failures(relative_manifest))
    if failures:
        raise SystemExit("\n".join(failures))


def check_simulated_release() -> None:
    release = json.loads(
        (ROOT / "artifacts/releases/simulated-feed-control.json").read_text()
    )
    report_path = release_resource_path(
        ROOT, release["source_report"]["logical_key"]
    )
    report = json.loads(report_path.read_text())
    failures = []
    if sha256(report_path.read_bytes()).hexdigest() != release["source_report"]["sha256"]:
        failures.append("simulated release does not bind the launch report")
    if release["active_control_artifact"] != report["release_state"]["active_artifact"]:
        failures.append("simulated release active artifact differs from launch state")
    roles = (
        ("active_control_artifact", "active_artifact_collection"),
        ("rollback_artifact", "rollback_artifact_collection"),
    )
    for role, collection_role in roles:
        manifest = release[role]
        artifact_dir = release_resource_path(ROOT, release[collection_role])
        artifact = artifact_dir / manifest["artifact_file"]
        artifact_id = f"sha256:{sha256(artifact.read_bytes()).hexdigest()}"
        if artifact_id != manifest["artifact_id"]:
            failures.append(f"simulated release {role} hash mismatch")
    if failures:
        raise SystemExit("\n".join(failures))


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=capture)


def feature_campaign_requirements(feature, small, ablation) -> dict[str, bool]:
    decisions = tuple(
        {
            launch["launch_id"]: launch["decision"]
            for launch in report["launches"]
        }
        for report in (feature, small, ablation)
    )
    expected = (
        {
            "F-LR-001": "hold_unified_lt_uncertain",
            "F-LR-002": "pass_unified_lt_nonnegative",
            "F-LR-003": "pass_unified_lt_nonnegative",
            "F-LR-004": "reject_unified_lt_negative",
        },
        {
            "F-LR-005": "reject_unified_lt_negative",
            "F-LR-006": "hold_unified_lt_uncertain",
            "F-LR-007": "pass_unified_lt_nonnegative",
        },
        {
            "F-LR-008": "hold_unified_lt_uncertain",
            "F-LR-009": "hold_unified_lt_uncertain",
            "F-LR-010": "hold_unified_lt_uncertain",
            "F-LR-011": "hold_unified_lt_uncertain",
            "F-LR-012": "reject_unified_lt_negative",
        },
    )
    return {
        "feature_lr_isolated_decisions": decisions[0] == expected[0],
        "small_feature_lr_decisions": decisions[1] == expected[1],
        "local_ablation_lr_decisions": decisions[2] == expected[2],
        "feature_lr_last_accepted_control": feature["release_state"]["active_key"]
        == "basic__realtime__local_context",
        "small_feature_lr_active_control": small["release_state"]["active_key"]
        == "basic__realtime__local_context__category_hash",
        "local_ablation_preserves_active": all(
            launch["control"] == launch["promotion"]["prior_active_key"]
            and launch["removed_features"]
            for launch in ablation["launches"]
        )
        and ablation["release_state"]["active_key"]
        == "basic__realtime__local_context__category_hash",
        "feature_campaign_tensor_throughput": min(
            value["requests_per_second"]
            for report in (feature, small, ablation)
            for value in report["world_performance"].values()
        )
        > 2_000_000,
    }


def acceptance_requirements(
    result,
    training,
    generative,
    poi,
    poi_feed,
    surfaces,
    evolution,
    ab,
    tensor_launch,
    feature_launch,
    small_feature_launch,
    local_ablation_launch,
    request_dataset,
) -> dict[str, bool]:
    return {
        "full_slate_rate": result["full_slate_rate"] == 1.0,
        "unsafe_items": result["unsafe_items"] == 0,
        "duplicates": result["slates_with_duplicates"] == 0,
        "category_coverage": result["mean_categories_per_slate"] >= 4.0,
        "joined_examples": training["joiner"]["examples"] == 600,
        "online_model_updated": training["parameter_server"]["model_version"] > 0,
        "training_consistency": training["consistency"]["passed"],
        "semantic_ids_unique": generative["items"] == generative["unique_codes"],
        "generative_items_valid": generative["valid_generated"],
        "generative_recall_complete": generative["generated"] == 20,
        "poi_hard_negatives": poi["hard_negatives"] > 0,
        "poi_ranking_beats_baseline": poi["model_ndcg_at_3"]
        > poi["baseline_ndcg_at_3"],
        "poi_sparse_publish_task": poi["task_metrics"]["publish"]["positive_rate"]
        < 0.1,
        "poi_feed_extracted_from_main": poi_feed["anchored_impressions"]
        == poi_feed["examples"],
        "poi_feed_sparse_order": poi_feed["label_rates"]["order"] < 0.02,
        "poi_feed_full_path_consistency": poi_feed["consistency"]["passed"],
        "surface_models_learn": surfaces["all_tasks_above_random"],
        "surface_model_count": len(surfaces["surfaces"]) == 5,
        "model_evolution_count": len(evolution["ranking"]) == 8,
        "retrieval_evolution_count": len(evolution["retrieval"]["models"]) == 5,
        "ab_recovers_known_truth": ab["all_truth_covered"],
        "tensor_semantic_parity": tensor_launch["semantic_parity"]["passed"],
        "tensor_unified_lt_gate": tensor_launch["launch_decision"]
        == "pass_unified_lt_nonnegative"
        and tensor_launch["unified_lt_exchange"]["overall_nonnegative"],
        "tensor_model_throughput": tensor_launch["treatment"]["performance"][
            "requests_per_second"
        ]
        > 2_000_000,
        **feature_campaign_requirements(
            feature_launch, small_feature_launch, local_ablation_launch
        ),
        "request_candidate_dataset_closes": (
            request_dataset["one_exposure_per_request"]
            and request_dataset["candidate_decisions"]
            == request_dataset["mature_label_rows"]
        ),
    }


def main() -> None:
    check_structure()
    check_public_docs()
    check_report_manifests()
    check_visual_manifest()
    check_model_artifacts()
    check_simulated_release()
    run([sys.executable, "-m", "compileall", "-q", "fid_lab", "tests"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    benchmark = run([sys.executable, "-m", "fid_lab.online.benchmark"], capture=True)
    result = json.loads(benchmark.stdout)
    training_demo = run([sys.executable, "-m", "fid_lab.training.demo"], capture=True)
    training = json.loads(training_demo.stdout)
    generative_demo = run([sys.executable, "-m", "fid_lab.generative.demo"], capture=True)
    generative = json.loads(generative_demo.stdout)
    poi_demo = run([sys.executable, "-m", "fid_lab.poi_posting.demo"], capture=True)
    poi = json.loads(poi_demo.stdout)
    poi_feed_demo = run([sys.executable, "-m", "fid_lab.poi_feed.demo"], capture=True)
    poi_feed = json.loads(poi_feed_demo.stdout)
    surface_demo = run([sys.executable, "-m", "fid_lab.surfaces.demo"], capture=True)
    surfaces = json.loads(surface_demo.stdout)
    evolution_demo = run(
        [sys.executable, "-m", "fid_lab.evolution.evaluation.benchmark", "--profile", "ci"],
        capture=True,
    )
    evolution = json.loads(evolution_demo.stdout)
    ab_demo = run(
        [sys.executable, "-m", "fid_lab.evolution.cli.ab_demo", "--users", "200000"],
        capture=True,
    )
    ab = json.loads(ab_demo.stdout)
    tensor_launch = json.loads(
        (ROOT / "reports/launches/2026-08-23-tensor-artifact-v2-1m-gpu.json").read_text()
    )
    feature_launch = json.loads(
        (
            ROOT
            / "reports/launches/2026-08-23-feature-lr-sequential-1m-gpu.json"
        ).read_text()
    )
    small_feature_launch = json.loads(
        (
            ROOT / "reports/launches/2026-08-23-feature-lr-hash-split-1m-gpu.json"
        ).read_text()
    )
    local_ablation_launch = json.loads(
        (
            ROOT
            / "reports/launches/2026-08-23-feature-lr-local-ablation-1m-gpu.json"
        ).read_text()
    )
    stateful_feature = json.loads(
        (ROOT / "reports/launches/2026-08-23-feature-lr-stateful-500.json").read_text()
    )
    request_dataset = stateful_feature["joiner"]["request_candidate_dataset"]
    required = acceptance_requirements(
        result,
        training,
        generative,
        poi,
        poi_feed,
        surfaces,
        evolution,
        ab,
        tensor_launch,
        feature_launch,
        small_feature_launch,
        local_ablation_launch,
        request_dataset,
    )
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise SystemExit(f"acceptance failures: {', '.join(failed)}")
    print(json.dumps({"status": "PASS", "acceptance": required}, indent=2))


if __name__ == "__main__":
    main()
