"""CLI for one pre-registered sequential Feed Launch Review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..profile import STANDARD_FEED_PROFILE
from ..runtime_paths import RuntimePaths
from .launch import FeedLaunchSpec, initialize_canonical_runtime, run_feed_launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    paths = RuntimePaths.standard(
        STANDARD_FEED_PROFILE,
        args.runtime_root,
    )
    if args.initialize_only:
        result = {
            "checkpoint_id": initialize_canonical_runtime(
                paths,
                STANDARD_FEED_PROFILE,
                device=args.device,
            ),
            "runtime_root": str(paths.root),
            "profile_hash": STANDARD_FEED_PROFILE.profile_hash,
        }
    else:
        if args.plan is None:
            parser.error("--plan is required unless --initialize-only is used")
        result = run_feed_launch(
            paths,
            FeedLaunchSpec.load(args.plan),
            STANDARD_FEED_PROFILE,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
