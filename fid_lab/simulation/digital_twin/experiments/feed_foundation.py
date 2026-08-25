"""Launch Feed impression dedup before new-item exploration and model LRs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Literal, Mapping

import torch

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..contracts import ContentKind, Surface
from ..engine import AtomicSimulationKernel, ExperimentPlan
from ..learning.request_stream import FactualRequestStream
from ..platform import CascadePolicy
from .retrieval_ladder import (
    RetrievalLadderConfig,
    _analyze,
    _baseline_plan,
    _build_kernel,
    _decision,
    _sync,
)


@dataclass(frozen=True)
class FeedDedupLaunchConfig:
    checkpoint_root: str
    request_stream_root: str
    checkpoint_branch: str = "main"
    users: int = 20_000
    items: int = 500_000
    device: str = "cuda"
    seed: int = 809
    ticks_per_day: int = 8
    experiment_steps: int = 8
    control_fraction: float = 0.2
    treatment_fraction: float = 0.2
    minimum_triggered_users: int = 500
    maximum_attempts: int = 3
    dedup_ticks: int = 16
    dedup_mode: Literal["window", "session"] = "window"
    launch_id: str = "F-LR-008"
    cursor_key: str = "feed_dedup_launch"
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False

    def __post_init__(self) -> None:
        if self.dedup_mode not in {"window", "session"}:
            raise ValueError("dedup mode must be window or session")
        if not self.launch_id or not self.cursor_key:
            raise ValueError("launch id and cursor key are required")


def _runtime_config(config: FeedDedupLaunchConfig) -> RetrievalLadderConfig:
    return RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
        experiment_steps=config.experiment_steps,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        minimum_triggered_users=config.minimum_triggered_users,
    )


def _repeat_counts(
    trace,
    exposure_item: torch.Tensor,
    exposure_time: torch.Tensor,
    content_kind: torch.Tensor,
    window_ticks: int,
    session_start_time: torch.Tensor | None = None,
) -> dict[str, dict[str, int]]:
    prior_item = exposure_item[trace.user_id]
    prior_time = exposure_time[trace.user_id]
    recent = (
        (prior_item >= 0)
        & (trace.event_time[:, None] - prior_time >= 0)
        & (trace.event_time[:, None] - prior_time <= window_ticks)
    )
    if session_start_time is not None:
        recent &= prior_time >= session_start_time[trace.user_id, None]
    exposed = trace.exposed_item_id
    repeated = (
        (exposed[:, :, None] == prior_item[:, None, :])
        & recent[:, None, :]
        & (exposed[:, :, None] >= 0)
    ).any(dim=2)
    repeated &= content_kind[exposed.clamp_min(0)] == int(
        ContentKind.SHORT_VIDEO
    )
    result = {}
    for cell, name in ((0, "control"), (1, "treatment")):
        selected = (trace.experiment_cell == cell) & (
            trace.surface == int(Surface.FEED)
        )
        result[name] = {
            "exposures": int((exposed[selected] >= 0).sum()),
            "repeats": int(repeated[selected].sum()),
        }
    return result


def _merge_repeat_counts(
    left: dict[str, dict[str, int]],
    right: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int]]:
    if right is None:
        return left
    return {
        cell: {
            metric: value + int(right[cell][metric])
            for metric, value in metrics.items()
        }
        for cell, metrics in left.items()
    }


def _repeat_rates(counts: Mapping[str, Mapping[str, int]]) -> dict[str, float]:
    return {
        cell: values["repeats"] / max(values["exposures"], 1)
        for cell, values in counts.items()
    }


def _run_experiment_window(
    kernel: AtomicSimulationKernel,
    stream: FactualRequestStream,
    plan: ExperimentPlan,
    *,
    logical_time: int,
    steps: int,
    dedup_ticks: int,
    dedup_mode: str,
    transaction_id: str,
) -> tuple[int, dict[str, dict[str, int]], tuple[object, ...]]:
    repeat_counts = {
        "control": {"exposures": 0, "repeats": 0},
        "treatment": {"exposures": 0, "repeats": 0},
    }
    staged_refs = []
    for _ in range(steps):
        before = kernel.platform.projection.snapshot().state
        tick = kernel.step(logical_time, plan)
        staged_refs.append(stream.stage(
            transaction_id,
            tick,
            kernel.platform.projection.snapshot(),
            kernel.world.manifest(),
        ))
        counts = _repeat_counts(
            tick.candidate_trace,
            before.user_feed_exposure_item,
            before.user_feed_exposure_time,
            kernel.world.catalog.content_kind,
            dedup_ticks,
            (
                kernel.platform.projection.state.user_session_start_time
                if dedup_mode == "session" else None
            ),
        )
        repeat_counts = _merge_repeat_counts(counts, repeat_counts)
        logical_time += 1
    return logical_time, repeat_counts, tuple(staged_refs)


def _review_window(
    *,
    events,
    users: int,
    minimum_triggered_users: int,
    maximum_attempts: int,
    attempt: int,
) -> tuple[dict[str, float], dict[str, int], str, str]:
    metrics, sample = _analyze([events], users)
    decision, reason = _decision(metrics, sample, minimum_triggered_users)
    if decision == "hold" and attempt >= maximum_attempts:
        return (
            metrics,
            sample,
            "stop_inconclusive",
            "maximum review windows reached without conclusive lift",
        )
    return metrics, sample, decision, reason


def _updated_learning_cursors(
    previous: Mapping[str, object],
    *,
    branch_name: str,
    stream: FactualRequestStream,
    logical_time: int,
    attempt: int,
    start_time: int,
    repeat_counts: Mapping[str, Mapping[str, int]],
    reviews: list[dict[str, object]],
    completed: bool,
    decision: str,
    cursor_key: str,
) -> dict[str, object]:
    cursors = dict(previous)
    cursors["factual_request_stream"] = {
        "branch": branch_name,
        "stream_sha256": stream.stream_sha256,
        "last_collected_time": logical_time - 1,
        "partitions": len(stream.refs(training=True)),
    }
    cursors[cursor_key] = {
        "attempt": attempt,
        "analysis_start_time": start_time,
        "repeat_counts": repeat_counts,
        "reviews": reviews,
        "completed": completed,
        "decision": decision,
    }
    return cursors


def _dedup_treatment_plan(
    active: CascadePolicy,
    config: FeedDedupLaunchConfig,
) -> tuple[CascadePolicy, ExperimentPlan]:
    treatment_updates = {
        "name": f"feed-{config.dedup_mode}-dedup-v1",
        "recall_version_id": max(active.recall_version_id + 1, 8),
    }
    if config.dedup_mode == "session":
        treatment_updates["feed_session_dedup"] = True
    else:
        treatment_updates["feed_exposure_dedup_ticks"] = config.dedup_ticks
    treatment = replace(active, **treatment_updates)
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=treatment,
        experiment_seed=config.seed + 800,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )
    return treatment, plan


def _active_cursor(
    learning_cursors: Mapping[str, object], cursor_key: str,
) -> dict[str, object]:
    cursor = dict(learning_cursors.get(cursor_key, {}))
    if cursor.get("completed"):
        raise ValueError("Feed dedup launch is already complete")
    return cursor


def _ensure_exposure_ledger(kernel, logical_time: int) -> None:
    if not int(kernel.platform.projection.state.user_feed_exposure_cursor.sum()):
        kernel.platform.projection.rebuild_exposures(
            kernel.event_log.read(ingested_through=logical_time),
        )


def _save_launch_checkpoint(
    store, registry, branch, kernel, logical_time, plan, cursors, stream,
    restore_time,
):
    try:
        checkpoint = store.save(
            kernel,
            logical_time,
            plan,
            parent_checkpoint_id=branch.head_checkpoint_id,
            learning_cursors=cursors,
        )
        registry.advance(
            branch.name,
            checkpoint.checkpoint_id,
            expected_head_checkpoint_id=branch.head_checkpoint_id,
        )
        return checkpoint
    except Exception:
        stream.reconcile_through(restore_time)
        raise


def run_feed_dedup_launch(config: FeedDedupLaunchConfig) -> dict[str, object]:
    device, kernel = _build_kernel(_runtime_config(config))
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("Feed foundation LR requires the factual branch")
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=config.allow_additive_runtime_migration,
    )
    if not isinstance(restored.experiment, ExperimentPlan):
        raise ValueError("Feed foundation LR requires a non-layered experiment")
    cursor = _active_cursor(restored.learning_cursors, config.cursor_key)
    _ensure_exposure_ledger(kernel, restored.ref.logical_time)
    active = restored.experiment.policies[-1]
    if not isinstance(active, CascadePolicy):
        raise ValueError("Feed foundation control is not a cascade policy")
    treatment, plan = _dedup_treatment_plan(active, config)
    stream = FactualRequestStream(
        Path(config.request_stream_root) / branch.name, branch,
    )
    orphaned = stream.reconcile_through(restored.ref.logical_time)
    logical_time = restored.ref.logical_time + 1
    start_time = int(cursor.get("analysis_start_time", logical_time))
    attempt = int(cursor.get("attempt", 0)) + 1
    transaction_id = (
        f"{config.launch_id}-attempt-{attempt}-"
        f"{branch.head_checkpoint_id[:12]}"
    )
    _sync(device)
    started = time.perf_counter()
    try:
        logical_time, repeat_counts, staged_refs = _run_experiment_window(
            kernel,
            stream,
            plan,
            logical_time=logical_time,
            steps=config.experiment_steps,
            dedup_ticks=config.dedup_ticks,
            dedup_mode=config.dedup_mode,
            transaction_id=transaction_id,
        )
        stream.commit_staged(transaction_id, staged_refs)
    except Exception:
        stream.abort_staged(transaction_id)
        stream.reconcile_through(restored.ref.logical_time)
        raise
    repeat_counts = _merge_repeat_counts(
        repeat_counts,
        cursor.get("repeat_counts"),
    )
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    events = events.select(events.ingest_time >= start_time)
    metrics, sample, decision, reason = _review_window(
        events=events,
        users=config.users,
        minimum_triggered_users=config.minimum_triggered_users,
        maximum_attempts=config.maximum_attempts,
        attempt=attempt,
    )
    promoted = decision == "promote"
    active_after = treatment if promoted else active
    reviews = [*cursor.get("reviews", []), {
        "launch_review": config.launch_id,
        "attempt": attempt,
        "analysis_start_time": start_time,
        "analysis_end_time": logical_time - 1,
        "changed_owner": f"Feed {config.dedup_mode} dedup only",
        "dedup_mode": config.dedup_mode,
        "dedup_ticks": config.dedup_ticks,
        "sample": sample,
        "metrics_per_triggered_user": metrics,
        "repeat_counts": repeat_counts,
        "repeat_rate": _repeat_rates(repeat_counts),
        "decision": decision,
        "reason": reason,
        "promoted_to_next_baseline": promoted,
    }]
    completed = decision in {"promote", "reject", "stop_inconclusive"}
    learning_cursors = _updated_learning_cursors(
        restored.learning_cursors,
        branch_name=branch.name,
        stream=stream,
        logical_time=logical_time,
        attempt=attempt,
        start_time=start_time,
        repeat_counts=repeat_counts,
        reviews=reviews,
        completed=completed,
        decision=decision,
        cursor_key=config.cursor_key,
    )
    checkpoint_plan = (
        _baseline_plan(_runtime_config(config), active_after, 8_100)
        if completed else plan
    )
    checkpoint = _save_launch_checkpoint(
        store, registry, branch, kernel, logical_time - 1,
        checkpoint_plan, learning_cursors, stream, restored.ref.logical_time,
    )
    _sync(device)
    return {
        "schema": "feed-foundation-launch-review/v1",
        "quality_claim": "synthetic-world causal evidence only",
        "config": asdict(config),
        "resumed_from_checkpoint": branch.head_checkpoint_id,
        "final_checkpoint_id": checkpoint.checkpoint_id,
        "active_policy": active_after.name,
        "review": reviews[-1],
        "request_stream_sha256": stream.stream_sha256,
        "reconciled_orphan_partitions": len(orphaned),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--request-stream-root", required=True)
    parser.add_argument("--checkpoint-branch", default="main")
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--ticks-per-day", type=int, default=8)
    parser.add_argument("--experiment-steps", type=int, default=8)
    parser.add_argument("--dedup-ticks", type=int, default=16)
    parser.add_argument("--dedup-mode", choices=("window", "session"), default="window")
    parser.add_argument("--launch-id", default="F-LR-008")
    parser.add_argument("--cursor-key", default="feed_dedup_launch")
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument("--allow-additive-runtime-migration", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_feed_dedup_launch(FeedDedupLaunchConfig(
        checkpoint_root=args.checkpoint_root,
        request_stream_root=args.request_stream_root,
        checkpoint_branch=args.checkpoint_branch,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
        experiment_steps=args.experiment_steps,
        dedup_ticks=args.dedup_ticks,
        dedup_mode=args.dedup_mode,
        launch_id=args.launch_id,
        cursor_key=args.cursor_key,
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
