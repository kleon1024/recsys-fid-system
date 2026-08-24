"""Pool independent unified-serving GPU Launch Reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import (
    aggregate_composite_launches,
    aggregate_governance_launches,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.inputs]
    report = (
        aggregate_governance_launches(reports)
        if reports[0].get("schema") == "content-governance-launch-v1"
        else aggregate_composite_launches(reports)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "shadow_gates": report["shadow_gates"],
        "online_gates": report["online_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
