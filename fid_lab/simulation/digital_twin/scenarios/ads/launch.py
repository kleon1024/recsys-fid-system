"""Independent Ads auction and Feed-guardrail Launch Review."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import torch

from ...contracts import Surface
from ...engine import ExperimentPlan
from ...experiments.retrieval_ladder import _sync
from ...platform import CascadePolicy
from ..launch_runtime import open_factual_launch, publish_factual_launch
from .audit import audit_ads_market
from .metrics import (
    ads_decision,
    ads_metrics,
    ads_trace_counts,
    merge_trace_counts,
)


@dataclass(frozen=True)
class AdsLaunchConfig:
    checkpoint_root: str
    request_stream_root: str
    checkpoint_branch: str = "main"
    users: int = 20_000
    items: int = 500_000
    device: str = "cuda"
    seed: int = 809
    ticks_per_day: int = 8
    experiment_steps: int = 32
    control_fraction: float = 0.2
    treatment_fraction: float = 0.2
    minimum_triggered_users: int = 500
    maximum_attempts: int = 5
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False
    launch_id: str = "A-LR-001"
    cursor_key: str = "ads_auction_launch_v1"


def _policies(active: CascadePolicy, config: AdsLaunchConfig):
    business_routes = tuple(
        route for route in active.enabled_business_routes
        if route != "ads_auction"
    )
    control = replace(
        active,
        name="ads-disabled-control",
        enabled_business_routes=business_routes,
    )
    treatment = replace(
        active,
        name="ads-budget-auction-v1",
        recall_version_id=max(active.recall_version_id + 1, 15),
        mix_version_id=max(active.mix_version_id + 1, 15),
        enabled_business_routes=(*business_routes, "ads_auction"),
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=config.seed + 15_000,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )
    return control, treatment, plan


def _stable_plan(active: CascadePolicy, config: AdsLaunchConfig):
    return ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=active,
        experiment_seed=config.seed + 15_101,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )


def _run_window(runtime, plan, logical_time, config, transaction_id, counts):
    staged_refs = []
    for _ in range(config.experiment_steps):
        tick = runtime.kernel.step(logical_time, plan)
        staged_refs.append(runtime.stream.stage(
            transaction_id,
            tick,
            runtime.kernel.platform.projection.snapshot(),
            runtime.kernel.world.manifest(),
        ))
        counts = merge_trace_counts(
            ads_trace_counts(
                tick.candidate_trace, runtime.kernel.world.catalog.content_kind,
            ),
            counts,
        )
        logical_time += 1
    return logical_time, counts, tuple(staged_refs)


def run_ads_launch(config: AdsLaunchConfig) -> dict[str, object]:
    runtime = open_factual_launch(config)
    restored = runtime.restored
    if not isinstance(restored.experiment, ExperimentPlan):
        raise ValueError("Ads market LR requires a stable non-layered baseline")
    active = restored.experiment.policies[-1]
    if not isinstance(active, CascadePolicy):
        raise TypeError("Ads market LR requires a cascade policy")
    cursor = dict(restored.learning_cursors.get(config.cursor_key, {}))
    if cursor.get("completed"):
        raise ValueError("Ads auction launch is already complete")
    control, treatment, plan = _policies(active, config)
    logical_time = restored.ref.logical_time + 1
    start_time = int(cursor.get("analysis_start_time", logical_time))
    attempt = int(cursor.get("attempt", 0)) + 1
    transaction_id = (
        f"{config.launch_id}-attempt-{attempt}-"
        f"{runtime.branch.head_checkpoint_id[:12]}"
    )
    counts = {
        "control_requests": 0,
        "treatment_requests": 0,
        "control_auction_candidates": 0,
        "treatment_auction_candidates": 0,
        "control_ad_exposures": 0,
        "treatment_ad_exposures": 0,
    }
    _sync(runtime.device)
    started = time.perf_counter()
    try:
        logical_time, counts, staged_refs = _run_window(
            runtime, plan, logical_time, config, transaction_id, counts,
        )
    except Exception:
        runtime.stream.abort_staged(transaction_id)
        raise
    counts = merge_trace_counts(counts, cursor.get("trace_counts"))
    all_events = runtime.kernel.event_log.read(
        ingested_through=logical_time - 1,
    )
    window_events = all_events.select(all_events.ingest_time >= start_time)
    metrics, sample = ads_metrics(window_events, config.users)
    audit = audit_ads_market(all_events, start_time=start_time)
    decision, reason = ads_decision(
        metrics, sample, audit, config.minimum_triggered_users,
    )
    if decision == "hold" and attempt >= config.maximum_attempts:
        decision, reason = "stop_inconclusive", "maximum review windows reached"
    completed = decision in {
        "promote", "reject", "stop_inconclusive", "no_support",
    }
    active_after = treatment if decision == "promote" else control
    checkpoint_plan = _stable_plan(active_after, config) if completed else plan
    review = {
        "launch_review": config.launch_id,
        "attempt": attempt,
        "analysis_window": [start_time, logical_time - 1],
        "pixel_mature_through": (
            logical_time - 1 - runtime.kernel.world.max_reporting_lag
        ),
        "changed_owner": "Ads auction, pacing and one-slot load only",
        "sample": sample,
        "trace_counts": counts,
        "market_audit": asdict(audit),
        "metrics_per_triggered_user": metrics,
        "decision": decision,
        "reason": reason,
    }
    cursors = dict(restored.learning_cursors)
    cursors[config.cursor_key] = {
        "attempt": attempt,
        "analysis_start_time": start_time,
        "trace_counts": counts,
        "reviews": [*cursor.get("reviews", []), review],
        "completed": completed,
        "decision": decision,
    }
    checkpoint = publish_factual_launch(
        runtime,
        transaction_id=transaction_id,
        staged_refs=staged_refs,
        logical_time=logical_time - 1,
        plan=checkpoint_plan,
        learning_cursors=cursors,
    )
    _sync(runtime.device)
    return {
        "schema": "ads-auction-launch-review/v1",
        "quality_claim": "synthetic-world causal evidence only",
        "config": asdict(config),
        "resumed_from_checkpoint": runtime.branch.head_checkpoint_id,
        "final_checkpoint_id": checkpoint.checkpoint_id,
        "review": review,
        "request_stream_sha256": runtime.stream.stream_sha256,
        "reconciled_orphan_partitions": len(runtime.reconciled_orphans),
        "repaired_history_chronology_violations": (
            runtime.repaired_history_violations
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(runtime.device) / 2**30
            if runtime.device.type == "cuda" else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--request-stream-root", required=True)
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-steps", type=int, default=32)
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument(
        "--allow-additive-runtime-migration", action="store_true",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_ads_launch(AdsLaunchConfig(
        checkpoint_root=args.checkpoint_root,
        request_stream_root=args.request_stream_root,
        users=args.users,
        items=args.items,
        device=args.device,
        experiment_steps=args.experiment_steps,
        allow_code_migration=args.allow_code_migration,
        allow_additive_runtime_migration=args.allow_additive_runtime_migration,
    ))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
