"""GPU batch-size Pareto benchmark with batch-invariance checks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ..tensor_engine import PERSONALIZED, TensorFeedConfig, run_tensor_feed


METRICS = (
    "stay_per_exposure",
    "long_view_rate",
    "lt_value_per_user",
    "anchor_click_rate",
    "coarse_pass_fraction",
)


def run_batch_benchmark(config, batch_sizes, memory_budget_bytes):
    runs = []
    reference = None
    for batch_users in batch_sizes:
        report = run_tensor_feed(
            replace(config, batch_users=batch_users), PERSONALIZED
        )
        metrics = {name: report["metrics"][name] for name in METRICS}
        stages = report["candidate_graph"]["stage_attribution"]
        if reference is None:
            reference = {"metrics": metrics, "stages": stages}
        runs.append({
            "batch_users": batch_users,
            "performance": report["performance"],
            "metrics": metrics,
            "stage_counts_equal": stages == reference["stages"],
            "max_metric_absolute_delta": max(
                abs(metrics[name] - reference["metrics"][name])
                for name in METRICS
            ),
        })
    eligible = [
        row for row in runs
        if row["performance"]["peak_gpu_memory_bytes"] <= memory_budget_bytes
    ]
    if not eligible:
        raise ValueError("no batch size fits the GPU memory budget")
    selected = max(
        eligible, key=lambda row: row["performance"]["requests_per_second"]
    )
    return {
        "schema": "tensor-batch-pareto-v1",
        "config": {
            "users": config.users,
            "steps": config.steps,
            "signal_version": config.signal_version,
            "candidate_graph_version": config.candidate_graph_version,
        },
        "memory_budget_bytes": memory_budget_bytes,
        "selected_batch_users": selected["batch_users"],
        "runs": runs,
        "invariant": (
            "Stage counts must be equal and metric deltas must remain below "
            "the floating-point reduction tolerance."
        ),
        "evidence_boundary": "Single RTX 4090 synthetic throughput benchmark.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--memory-budget-gib", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = TensorFeedConfig(
        users=args.users, steps=args.steps, device=args.device,
        signal_version="kuairand-calibrated-v3",
    )
    report = run_batch_benchmark(
        config, tuple(args.batch_sizes), int(args.memory_budget_gib * 2**30)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected_batch_users": report["selected_batch_users"],
        "candidate_graph": config.candidate_graph_version,
    }, indent=2))


if __name__ == "__main__":
    main()
