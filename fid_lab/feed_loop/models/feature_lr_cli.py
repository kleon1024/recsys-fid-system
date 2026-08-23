"""CLI for the isolated feature-group LR training suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .feature_lr import train_feature_lr_campaign, train_feature_lr_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=3_000)
    parser.add_argument("--items", type=int, default=8_000)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign")
    parser.add_argument("--base-report", type=Path)
    parser.add_argument("--base-artifact-dir", type=Path)
    args = parser.parse_args()
    if args.campaign:
        report = train_feature_lr_campaign(
            args.campaign,
            args.users,
            args.items,
            args.artifact_dir,
            base_report_path=args.base_report,
            base_artifact_dir=args.base_artifact_dir,
        )
    else:
        report = train_feature_lr_suite(
            args.users, args.items, args.artifact_dir
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "examples": report["examples"],
        "offline": {
            name: {
                "auc": value["auc"],
                "gauc": value["user_gauc"]["value"],
                "oracle_regret": value["candidate"]["oracle_regret"],
            }
            for name, value in report["offline"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
