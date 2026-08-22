"""Single local acceptance gate for structure, tests, and system behavior."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (ROOT / "fid_lab", ROOT / "tests")
PUBLIC_DOCS = (
    "README.md",
    "REQUEST_FOR_PROPOSAL.md",
    "BIDDER_RESPONSE_TEMPLATE.md",
    "ARCHITECTURE_VISUALS.md",
    "SECURITY.md",
    "POI_POSTING_MODEL.md",
    "PRODUCTION_MODEL_SUITE.md",
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


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=capture)


def main() -> None:
    check_structure()
    check_public_docs()
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
    required = {
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
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise SystemExit(f"acceptance failures: {', '.join(failed)}")
    print(json.dumps({"status": "PASS", "acceptance": required}, indent=2))


if __name__ == "__main__":
    main()
