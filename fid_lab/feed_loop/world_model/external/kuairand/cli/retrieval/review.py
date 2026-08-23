"""Aggregate retrieval seed reports into one Launch Review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...retrieval.review import build_retrieval_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_retrieval_review(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "active_retrieval_control": report["active_retrieval_control"],
        "launches": [
            {"treatment": row["treatment"], "decision": row["decision"]}
            for row in report["launches"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
