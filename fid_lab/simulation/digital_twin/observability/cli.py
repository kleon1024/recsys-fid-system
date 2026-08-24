"""Generate a deterministic v4 full-flow analytical fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .fixture import FullFlowFixtureConfig, build_full_flow_fixture
from .store import materialize_full_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=64)
    parser.add_argument("--items", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--route-k", type=int, default=6)
    parser.add_argument("--merged-k", type=int, default=24)
    parser.add_argument("--coarse-k", type=int, default=12)
    parser.add_argument("--fine-k", type=int, default=6)
    parser.add_argument("--expose-k", type=int, default=3)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--recall-negatives", type=int, default=4)
    parser.add_argument("--seed-failures", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    snapshot = build_full_flow_fixture(FullFlowFixtureConfig(
        users=args.users,
        items=args.items,
        device=args.device,
        route_k=args.route_k,
        merged_k=args.merged_k,
        coarse_k=args.coarse_k,
        fine_k=args.fine_k,
        expose_k=args.expose_k,
        history_length=args.history_length,
        recall_negatives=args.recall_negatives,
    ))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    built = time.perf_counter()
    manifest = materialize_full_flow(
        snapshot,
        args.output,
        seed_failures=args.seed_failures,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    completed = time.perf_counter()
    print(json.dumps({
        "manifest": manifest,
        "runtime": {
            "build_seconds": built - started,
            "materialize_seconds": completed - built,
            "total_seconds": completed - started,
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else 0.0
            ),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
