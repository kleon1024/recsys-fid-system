"""Independent Commerce inventory-eligibility Launch Review."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import torch

from ...checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ...contracts import Surface
from ...engine import ExperimentPlan
from ...experiments.layered import LayeredExperimentPlan, PolicyLayer
from ...experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
    _sync,
)
from ...learning.request_stream import FactualRequestStream
from ...platform import CascadePolicy
from ...profile import STANDARD_FEED_PROFILE
from .metrics import (
    commerce_decision,
    commerce_metrics,
    commerce_trace_counts,
    merge_counts,
)


@dataclass(frozen=True)
class CommerceLaunchConfig:
    checkpoint_root: str
    request_stream_root: str
    checkpoint_branch: str = "main"
    users: int = STANDARD_FEED_PROFILE.users
    items: int = STANDARD_FEED_PROFILE.items
    device: str = "cuda"
    seed: int = STANDARD_FEED_PROFILE.seed
    ticks_per_day: int = STANDARD_FEED_PROFILE.ticks_per_day
    experiment_steps: int = 32
    control_fraction: float = 0.2
    treatment_fraction: float = 0.2
    minimum_triggered_users: int = 200
    maximum_attempts: int = 5
    minimum_inventory: float = 0.0
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False
    launch_id: str = "C-LR-001"
    cursor_key: str = "commerce_inventory_launch_v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_inventory <= 1.0:
            raise ValueError("minimum inventory must be in [0, 1]")


def _runtime_config(config: CommerceLaunchConfig) -> RetrievalLadderConfig:
    return RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
        experiment_steps=config.experiment_steps,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
    )


def _active_policy(restored) -> CascadePolicy:
    if isinstance(restored.experiment, ExperimentPlan):
        return restored.experiment.policies[-1]
    if isinstance(restored.experiment, LayeredExperimentPlan):
        return restored.experiment.base_policy
    raise TypeError("unsupported experiment plan")


def _experiment(
    active: CascadePolicy, config: CommerceLaunchConfig,
) -> LayeredExperimentPlan:
    return LayeredExperimentPlan(active, (PolicyLayer(
        name="commerce-inventory-eligibility",
        salt=config.seed + 12_000,
        changes=(
            {"commerce_min_inventory": config.minimum_inventory}
            if config.minimum_inventory > 0.0
            else {"commerce_require_inventory": True}
        ),
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.COMMERCE),),
    ),))


def _stable_plan(
    active: CascadePolicy, config: CommerceLaunchConfig,
) -> ExperimentPlan:
    return ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=active,
        experiment_seed=config.seed + 12_101,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.COMMERCE),),
    )


def _run_staged_window(
    kernel, stream, plan, logical_time, config, transaction_id, counts,
):
    staged_refs = []
    for _ in range(config.experiment_steps):
        before = kernel.platform.projection.snapshot().state
        tick = kernel.step(logical_time, plan)
        if tick.candidate_trace is None or tick.request_context is None:
            logical_time += 1
            continue
        staged_refs.append(stream.stage(
            transaction_id,
            tick,
            kernel.platform.projection.snapshot(),
            kernel.world.manifest(),
        ))
        counts = merge_counts(
            commerce_trace_counts(
                tick.candidate_trace, before, config.minimum_inventory,
            ),
            counts,
        )
        logical_time += 1
    return logical_time, counts, tuple(staged_refs)


def _repair_request_history(kernel) -> int:
    violations = kernel.platform.projection.history_chronology_violations()
    if violations:
        kernel.platform.projection.rebuild_history(kernel.event_log.partitions())
    return violations


def run_commerce_launch(config: CommerceLaunchConfig) -> dict[str, object]:
    device, kernel = _build_kernel(_runtime_config(config))
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=config.allow_additive_runtime_migration,
    )
    history_violations = _repair_request_history(kernel)
    cursor = dict(restored.learning_cursors.get(config.cursor_key, {}))
    if cursor.get("completed"):
        raise ValueError("Commerce inventory launch is already complete")
    active = _active_policy(restored)
    plan = _experiment(active, config)
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
    counts = {
        "control_requests": 0,
        "treatment_requests": 0,
        "control_out_of_stock_product_exposures": 0,
        "treatment_out_of_stock_product_exposures": 0,
    }
    _sync(device)
    started = time.perf_counter()
    try:
        logical_time, counts, staged_refs = _run_staged_window(
            kernel, stream, plan, logical_time, config, transaction_id, counts,
        )
    except Exception:
        stream.abort_staged(transaction_id)
        raise
    counts = merge_counts(counts, cursor.get("counts"))
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    events = events.select(events.ingest_time >= start_time)
    metrics, sample = commerce_metrics(events, config.users)
    decision, reason = commerce_decision(
        metrics, sample, counts, config.minimum_triggered_users,
    )
    if decision == "hold" and attempt >= config.maximum_attempts:
        decision, reason = "stop_inconclusive", "maximum review windows reached"
    completed = decision in {
        "promote", "reject", "stop_inconclusive", "no_support",
    }
    treatment = replace(
        active,
        name="commerce-inventory-eligible-v1",
        commerce_require_inventory=(config.minimum_inventory <= 0.0),
        commerce_min_inventory=config.minimum_inventory,
    )
    active_after = treatment if decision == "promote" else active
    checkpoint_plan = (
        _stable_plan(active_after, config) if completed else plan
    )
    review = {
        "launch_review": config.launch_id,
        "attempt": attempt,
        "analysis_window": [start_time, logical_time - 1],
        "sample": sample,
        "traffic_counts": counts,
        "metrics_per_triggered_user": metrics,
        "decision": decision,
        "reason": reason,
    }
    cursors = dict(restored.learning_cursors)
    cursors[config.cursor_key] = {
        "attempt": attempt,
        "analysis_start_time": start_time,
        "counts": counts,
        "reviews": [*cursor.get("reviews", []), review],
        "completed": completed,
        "decision": decision,
    }
    try:
        stream.commit_staged(transaction_id, tuple(staged_refs))
        checkpoint = store.save(
            kernel,
            logical_time - 1,
            checkpoint_plan,
            parent_checkpoint_id=branch.head_checkpoint_id,
            learning_cursors=cursors,
        )
        registry.advance(
            branch.name,
            checkpoint.checkpoint_id,
            expected_head_checkpoint_id=branch.head_checkpoint_id,
        )
    except Exception:
        stream.abort_staged(transaction_id)
        stream.reconcile_through(restored.ref.logical_time)
        raise
    _sync(device)
    return {
        "schema": "commerce-inventory-launch-review/v1",
        "quality_claim": "synthetic-world causal evidence only",
        "config": asdict(config),
        "resumed_from_checkpoint": branch.head_checkpoint_id,
        "final_checkpoint_id": checkpoint.checkpoint_id,
        "review": review,
        "request_stream_sha256": stream.stream_sha256,
        "reconciled_orphan_partitions": len(orphaned),
        "repaired_history_chronology_violations": history_violations,
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
    parser.add_argument("--users", type=int, default=STANDARD_FEED_PROFILE.users)
    parser.add_argument("--items", type=int, default=STANDARD_FEED_PROFILE.items)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=STANDARD_FEED_PROFILE.seed)
    parser.add_argument(
        "--ticks-per-day", type=int,
        default=STANDARD_FEED_PROFILE.ticks_per_day,
    )
    parser.add_argument("--experiment-steps", type=int, default=32)
    parser.add_argument("--minimum-inventory", type=float, default=0.0)
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument(
        "--allow-additive-runtime-migration", action="store_true",
    )
    parser.add_argument("--launch-id", default="C-LR-001")
    parser.add_argument("--cursor-key", default="commerce_inventory_launch_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_commerce_launch(CommerceLaunchConfig(
        checkpoint_root=args.checkpoint_root,
        request_stream_root=args.request_stream_root,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
        experiment_steps=args.experiment_steps,
        minimum_inventory=args.minimum_inventory,
        allow_code_migration=args.allow_code_migration,
        allow_additive_runtime_migration=args.allow_additive_runtime_migration,
        launch_id=args.launch_id,
        cursor_key=args.cursor_key,
    ))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
