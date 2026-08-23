"""Publish the composite V4 simulator authority and its review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .authority import build_composite_world_review, build_world_release


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    review = build_composite_world_review(args.root)
    _write(args.review, review)
    release = build_world_release(args.review)
    _write(args.release, release)
    print(json.dumps({
        "decision": review["decision"],
        "feed_status": review["components"]["feed_behavior"]["status"],
        "full_world_status": review["components"]["unified_neural_scm"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
