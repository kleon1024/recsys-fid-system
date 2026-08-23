"""Run the external retrieval ladder and fixed-ranker shadow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ...launch.contracts import stream_sha256
from ...retrieval.ladder import RetrievalConfig, run_retrieval_ladder
from ...retrieval.shadow import run_retrieval_shadow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fine-artifact", type=Path, required=True)
    parser.add_argument("--world-artifact", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--fine-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--shadow-batch-size", type=int, default=128)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--evaluation-seed", type=int, default=20260824)
    args = parser.parse_args()
    config = RetrievalConfig(
        top_k=args.top_k, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed,
        evaluation_seed=args.evaluation_seed,
    )
    ladder = run_retrieval_ladder(
        args.dataset_dir, args.artifact_dir, config, args.device,
        args.max_train_rows,
    )
    shadow = run_retrieval_shadow(
        ladder, args.dataset_dir, args.fine_artifact, args.world_artifact,
        args.benchmark_report, args.fine_report, device=args.device,
        batch_size=args.shadow_batch_size,
    )
    candidate_path = args.artifact_dir / "evaluation_candidates.pt"
    torch.save({
        "candidate_sets": ladder.pop("candidate_sets"),
        "test_rows": ladder.pop("test_rows"),
        "target": ladder.pop("target"),
    }, candidate_path)
    report = {
        **ladder,
        "shadow": shadow,
        "candidate_artifact": {
            "sha256": stream_sha256(candidate_path),
            "format": "torch",
        },
        "decision": (
            "hold_aggregate_review_pending"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "models": report["models"],
        "frozen_retrieval_control": shadow["frozen_retrieval_control"],
        "provisional_passes": shadow["provisional_passes"],
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
