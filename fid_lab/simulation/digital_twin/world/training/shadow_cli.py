"""CLI for factual-cascade NeuralSCM authority shadow replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .shadow import AuthorityShadowConfig, run_authority_shadow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=200_000)
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=4_096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_authority_shadow(
        args.artifact_dir,
        AuthorityShadowConfig(
            users=args.users, items=args.items, ticks=args.ticks,
            member_index=args.member_index,
            inference_batch_size=args.inference_batch_size,
            device=args.device,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": report["decision"], "gates": report["gates"],
        "support_rate": report["first_run"]["support_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
