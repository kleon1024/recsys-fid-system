"""Single local acceptance gate for structure, tests, and system behavior."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

from .feed_loop.models.artifact import feature_schema_hash
from .feed_loop.scale.tensor_runtime.contracts import CANDIDATE_GRAPH_VERSION
from .launches.experiment_protocol import load_experiment_plan
from .simulation.environment import FEATURE_NAMES


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
    "docs/architecture/multi-surface-digital-twin.md",
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
        "reports/calibration/MANIFEST.sha256",
        "reports/world-model/v4/MANIFEST.sha256",
        "reports/training/MANIFEST.sha256",
        "reports/datasets/MANIFEST.sha256",
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


def check_experiment_plans() -> None:
    seen = set()
    for path in sorted((ROOT / "experiments/plans").glob("*.json")):
        plan = load_experiment_plan(path, ROOT)
        identity = (plan.launch_id, plan.phase.value)
        if identity in seen:
            raise SystemExit(f"duplicate experiment plan phase: {identity}")
        seen.add(identity)
    if not seen:
        raise SystemExit("no registered experiment plans")


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
        "artifacts/models/feature-lr-v5-intent-trigger/MANIFEST.sha256",
        "artifacts/models/v3-model-ladder/MANIFEST.sha256",
        "artifacts/models/poi-posting-request-v1/MANIFEST.sha256",
        "artifacts/models/poi-posting-v4/MANIFEST.sha256",
        "artifacts/models/feed-posting-request-v1/MANIFEST.sha256",
        "artifacts/models/feed-posting-v42/MANIFEST.sha256",
        "artifacts/models/feed-posting-v43-esmm/MANIFEST.sha256",
        "artifacts/models/local-search-request-v1/MANIFEST.sha256",
        "artifacts/models/poi-detail-request-v1/MANIFEST.sha256",
        "artifacts/models/poi-distribution-v4/MANIFEST.sha256",
        "artifacts/models/poi-retrieval-v4-poi-only/MANIFEST.sha256",
        "artifacts/models/shared-retrieval-v4-sequence-skew/MANIFEST.sha256",
        "artifacts/models/shared-retrieval-v4-aligned/MANIFEST.sha256",
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
    failures = []
    if release.get("schema_version") != "simulated-feed-authority-v3":
        failures.append("simulated release is not the V3 authority")

    def verify(resource, label):
        path = ROOT / resource["path"]
        if not path.exists():
            failures.append(f"simulated release missing {label}: {resource['path']}")
        elif sha256(path.read_bytes()).hexdigest() != resource["sha256"]:
            failures.append(f"simulated release {label} hash mismatch")

    verify(release["source_report"], "source report")

    def verify_bundle(bundle, bundle_id, label):
        for index, resource in enumerate(bundle["model"]["sources"]):
            verify(resource, f"{label} model source {index}")
        if "artifact" in bundle["model"]:
            verify(bundle["model"]["artifact"], f"{label} model artifact")
            if bundle["model"]["artifact_id"] != (
                f"sha256:{bundle['model']['artifact']['sha256']}"
            ):
                failures.append(f"{label} model artifact id mismatch")
        verify(bundle["feature"]["source"], f"{label} feature source")
        for kind in ("index", "behavior"):
            for index, resource in enumerate(bundle[kind]["sources"]):
                verify(resource, f"{label} {kind} source {index}")
        encoded = json.dumps(
            bundle, sort_keys=True, separators=(",", ":")
        ).encode()
        if bundle_id != f"sha256:{sha256(encoded).hexdigest()}":
            failures.append(f"simulated release {label} bundle id mismatch")
        if bundle["feature"]["schema_sha256"] != feature_schema_hash():
            failures.append(f"simulated release {label} feature schema mismatch")
        if bundle["feature"]["dense_fields"] != len(FEATURE_NAMES):
            failures.append(f"simulated release {label} feature width mismatch")

    bundle = release["active_bundle"]
    verify_bundle(bundle, release["active_bundle_id"], "active")
    rollback = release.get("rollback_bundle")
    if rollback is not None:
        verify_bundle(rollback, release["rollback_bundle_id"], "rollback")
    report = json.loads((ROOT / release["source_report"]["path"]).read_text())
    if "release_state" in report:
        if report["release_state"]["active_key"] != release["active_control_key"]:
            failures.append("authority active key differs from Launch Review")
        if report["release_state"]["active_artifact"] != (
            bundle["model"].get("model_manifest")
        ):
            failures.append("authority active artifact differs from Launch Review")
    dataset = release["dataset"]
    logging_bundle_id = dataset.get("logging_bundle_id")
    logging_bundle = dataset.get("logging_bundle")
    if dataset["authority_bundle_id"] != logging_bundle_id:
        failures.append("V3 request log is not bound to its logging bundle")
    if logging_bundle is None:
        failures.append("V3 request log is missing its historical logging bundle")
    else:
        encoded = json.dumps(
            logging_bundle, sort_keys=True, separators=(",", ":")
        ).encode()
        if logging_bundle_id != f"sha256:{sha256(encoded).hexdigest()}":
            failures.append("V3 historical logging bundle id mismatch")
    public_dataset = json.loads(
        (ROOT / "reports/datasets/2026-08-23-v3-request-log-manifest.json").read_text()
    )
    for field in (
        "authority_bundle_id", "label_names", "evaluation_value_names",
        "training_contract", "tables",
    ):
        if dataset[field] != public_dataset[field]:
            failures.append(f"V3 request log manifest differs at {field}")
    for historical in release["historical_releases"]:
        verify(historical, f"historical {historical['epoch']} release")
    if failures:
        raise SystemExit("\n".join(failures))


def check_simulator_world_release() -> None:
    release_path = ROOT / "artifacts/releases/simulator-world.json"
    review_path = ROOT / "reports/world-model/v4/composite-launch-review.json"
    release = json.loads(release_path.read_text())
    review = json.loads(review_path.read_text())
    failures = []
    if release.get("schema") != "composite-simulator-world-authority-v1":
        failures.append("simulator world release schema mismatch")
    if release.get("source_review_sha256") != sha256(
        review_path.read_bytes()
    ).hexdigest():
        failures.append("simulator world release is not bound to its review")
    if review.get("decision") != "promote_feed_local_and_supply_kernels":
        failures.append("simulator world review did not accept task kernels")
    feed = review.get("components", {}).get("feed_behavior", {})
    active_feed = release.get("active_components", {}).get("feed_behavior", {})
    if feed.get("status") != "eligible_simulator_authority":
        failures.append("Feed kernel is not eligible for simulator authority")
    if active_feed.get("policy_artifact_sha256") != feed.get(
        "policy_artifact_sha256"
    ):
        failures.append("active Feed policy kernel differs from the accepted review")
    if active_feed.get("response_world_artifact_sha256") != feed.get(
        "response_world_artifact_sha256"
    ):
        failures.append("active Feed response world differs from the accepted review")
    for field in ("catalog_sha256", "profile_sha256"):
        if active_feed.get(field) != feed.get(field):
            failures.append(f"active Feed {field} differs from the accepted review")
    local = review.get("components", {}).get("local_response", {})
    active_local = release.get("active_components", {}).get("local_response", {})
    if local.get("status") != "eligible_simulator_authority":
        failures.append("Local kernel is not eligible for simulator authority")
    if active_local.get("artifact_sha256") != local.get("artifact_sha256"):
        failures.append("active Local kernel differs from the accepted review")
    if active_local.get("retrieval_artifact_sha256") != local.get(
        "retrieval_artifact_sha256"
    ):
        failures.append("active retrieval kernel differs from the accepted review")
    supply = review.get("components", {}).get("supply_response", {})
    active_supply = release.get("active_components", {}).get("supply_response", {})
    if supply.get("status") != "eligible_simulator_authority":
        failures.append("Supply kernel is not eligible for simulator authority")
    if active_supply.get("artifact_sha256") != supply.get("artifact_sha256"):
        failures.append("active Supply kernel differs from the accepted review")
    if release.get("production_readiness") != "simulator_only":
        failures.append("simulator world release overstates production readiness")
    if failures:
        raise SystemExit("\n".join(failures))


def _check_simulated_surface_release(
    relative, schema, label, expected_readiness, allow_stale_sources=False,
) -> None:
    release = json.loads((ROOT / relative).read_text())
    failures = []
    if release.get("schema") != schema:
        failures.append(f"{label} release schema mismatch")
    bundle = release["active_bundle"]
    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    if release["active_bundle_id"] != f"sha256:{sha256(encoded).hexdigest()}":
        failures.append(f"{label} bundle id mismatch")
    stale = release.get("status") == "stale_retrain_required"
    if stale and not allow_stale_sources:
        failures.append(f"{label} unexpectedly declares a stale authority")
    resources = [bundle["model_artifact"], release["source_report"]]
    if release.get("powered_ab_report") is not None:
        resources.append(release["powered_ab_report"])
    if not stale:
        resources.extend(bundle["sources"])
        resources.extend(bundle.get("evidence_reports", []))
    if "retrieval_artifact" in bundle:
        resources.append(bundle["retrieval_artifact"])
    if "training_dataset" in bundle:
        resources.append(bundle["training_dataset"])
    for resource in resources:
        path = ROOT / resource["path"]
        if not path.exists():
            failures.append(f"{label} resource missing: {resource['path']}")
        elif sha256(path.read_bytes()).hexdigest() != resource["sha256"]:
            failures.append(f"{label} resource hash mismatch: {resource['path']}")
    if release["production_readiness"] != expected_readiness:
        failures.append(f"{label} release overclaims production readiness")
    if failures:
        raise SystemExit("\n".join(failures))


def check_simulated_surface_releases() -> None:
    _check_simulated_surface_release(
        "artifacts/releases/simulated-poi-posting-control.json",
        "simulated-poi-posting-authority-v2", "POI posting",
        "hold_external_creator_and_supply_validation",
    )
    _check_simulated_surface_release(
        "artifacts/releases/simulated-feed-posting-control.json",
        "simulated-feed-posting-authority-v2", "Feed posting",
        "hold_external_creator_and_supply_validation",
    )
    _check_simulated_surface_release(
        "artifacts/releases/simulated-local-search-control.json",
        "simulated-local-search-authority-v1", "Local Search",
        "hold_external_query_and_transaction_validation",
    )
    _check_simulated_surface_release(
        "artifacts/releases/simulated-poi-detail-control.json",
        "simulated-poi-detail-authority-v1", "POI Detail",
        "hold_external_page_transaction_and_review_validation",
    )
    _check_simulated_surface_release(
        "artifacts/releases/simulated-poi-distribution-v4.json",
        "simulated-poi-distribution-v4-authority-v1", "POI distribution V4",
        "simulator_only_external_local_validation_required",
    )


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


def digital_twin_requirements(
    digital_twin, trigger_launch, batch_scale
) -> dict[str, bool]:
    graph = digital_twin["control"]["candidate_graph"]
    attribution = graph["stage_attribution"]
    trigger_decisions = {
        launch["launch_id"]: launch["decision"]
        for launch in trigger_launch["launches"]
    }
    traces = (
        digital_twin["control"]["request_candidate_trace"],
        digital_twin["treatment"]["request_candidate_trace"],
    )
    return {
        "gpu_candidate_graph_closes": (
            graph["version"] == CANDIDATE_GRAPH_VERSION
            and sum(attribution.values()) == graph["requests"]
            and all(attribution[name] > 0 for name in (
                "recall_miss", "coarse_miss", "fine_rank_miss"
            ))
        ),
        "gpu_candidate_graph_has_real_attrition": digital_twin["control"][
            "metrics"
        ]["coarse_pass_fraction"] < 1.0,
        "gpu_request_trace_closes": all(
            trace["requests"] > 0
            and trace["candidate_rows"] == trace["requests"] * 48
            and len(trace["sha256"]) == 64
            for trace in traces
        ),
        "gpu_candidate_graph_throughput": min(
            digital_twin[world]["performance"]["requests_per_second"]
            for world in ("control", "treatment")
        ) > 1_000_000,
        "gpu_batch_scale_preserves_world": (
            batch_scale["selected_batch_users"] == 200_000
            and all(
                run["stage_counts_equal"]
                and run["max_metric_absolute_delta"] < 1e-6
                for run in batch_scale["runs"]
            )
        ),
        "triggered_feature_decisions": trigger_decisions == {
            "F-LR-013": "hold_unified_lt_uncertain",
            "F-LR-014": "hold_unified_lt_uncertain",
        },
        "triggered_feature_preserves_active": trigger_launch["release_state"][
            "active_key"
        ] == "basic__realtime__local_context__category_hash",
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
    digital_twin,
    trigger_launch,
    batch_scale,
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
        **digital_twin_requirements(digital_twin, trigger_launch, batch_scale),
        "request_candidate_dataset_closes": (
            request_dataset["one_exposure_per_request"]
            and request_dataset["candidate_decisions"]
            == request_dataset["mature_label_rows"]
        ),
    }


def run_code_gates() -> None:
    run([sys.executable, "-m", "compileall", "-q", "fid_lab", "tests"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/core/digital_twin",
    ])


def main() -> None:
    check_structure()
    check_public_docs()
    check_report_manifests()
    check_visual_manifest()
    check_experiment_plans()
    check_model_artifacts()
    check_simulated_release()
    check_simulator_world_release()
    check_simulated_surface_releases()
    run_code_gates()
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
    digital_twin = json.loads(
        (
            ROOT
            / "reports/launches/2026-08-24-feed-digital-twin-cascade-v3-1m-gpu.json"
        ).read_text()
    )
    trigger_launch = json.loads(
        (
            ROOT
            / "reports/launches/2026-08-23-feature-lr-intent-trigger-1m-gpu.json"
        ).read_text()
    )
    batch_scale = json.loads(
        (
            ROOT
            / "reports/benchmarks/2026-08-24-tensor-batch-pareto-cascade-v3.json"
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
        digital_twin,
        trigger_launch,
        batch_scale,
        request_dataset,
    )
    calibrated = json.loads(
        (
            ROOT
            / "reports/launches/2026-08-23-feed-calibrated-v3-1m-gpu.json"
        ).read_text()
    )
    alignment = calibrated["calibration"]["alignment"]
    required.update({
        "calibrated_v3_world": calibrated["config"]["signal_version"]
        == "kuairand-calibrated-v3",
        "calibrated_v3_marginals": max(
            abs(metric["relative_error"]) for metric in alignment.values()
        ) < 0.20,
        "calibrated_v3_holds_uncertain_launch": calibrated["decision"]
        == "hold_unified_lt_uncertain",
    })
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise SystemExit(f"acceptance failures: {', '.join(failed)}")
    print(json.dumps({"status": "PASS", "acceptance": required}, indent=2))


if __name__ == "__main__":
    main()
