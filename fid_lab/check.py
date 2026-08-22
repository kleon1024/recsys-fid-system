"""Single local acceptance gate for structure, tests, and system behavior."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (ROOT / "fid_lab", ROOT / "tests")


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


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=capture)


def main() -> None:
    check_structure()
    run([sys.executable, "-m", "compileall", "-q", "fid_lab", "tests"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    benchmark = run([sys.executable, "-m", "fid_lab.online.benchmark"], capture=True)
    result = json.loads(benchmark.stdout)
    required = {
        "full_slate_rate": result["full_slate_rate"] == 1.0,
        "unsafe_items": result["unsafe_items"] == 0,
        "duplicates": result["slates_with_duplicates"] == 0,
        "category_coverage": result["mean_categories_per_slate"] >= 4.0,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise SystemExit(f"acceptance failures: {', '.join(failed)}")
    print(json.dumps({"status": "PASS", "acceptance": required}, indent=2))


if __name__ == "__main__":
    main()
