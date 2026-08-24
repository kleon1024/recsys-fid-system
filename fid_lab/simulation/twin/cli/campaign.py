"""Run the default recall/coarse/fine/mix evolution campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..experimentation.campaign import run_launch_campaign
from ..profiles import PROFILE_OVERRIDES, load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_OVERRIDES), default="screen"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_profile(args.profile, args.device)
    report = run_launch_campaign(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "active_policy": report["active_policy"]["name"],
        "decisions": {
            row["launch_id"]: row["decision"] for row in report["launches"]
        },
        "stage_counts": report["stage_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
