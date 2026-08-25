"""Advance the accepted factual policy while persisting request-level facts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..engine import ExperimentPlan
from ..learning.request_stream import FactualRequestStream
from .layered import LayeredExperimentPlan
from .retrieval_ladder import RetrievalLadderConfig, _build_kernel, _sync


@dataclass(frozen=True)
class FactualCollectionConfig:
    checkpoint_root: str
    request_stream_root: str
    steps: int = 8
    checkpoint_branch: str = "main"
    users: int = 20_000
    items: int = 500_000
    device: str = "cuda"
    seed: int = 809
    ticks_per_day: int = 8
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.users <= 0 or self.items <= 0:
            raise ValueError("factual collection dimensions must be positive")
        if not self.checkpoint_root or not self.request_stream_root:
            raise ValueError("factual collection requires durable roots")


def _runtime_config(config: FactualCollectionConfig) -> RetrievalLadderConfig:
    return RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
        checkpoint_root=config.checkpoint_root,
        checkpoint_branch=config.checkpoint_branch,
    )


def collect_factual_requests(
    config: FactualCollectionConfig,
) -> dict[str, object]:
    device, kernel = _build_kernel(_runtime_config(config))
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("only the factual training branch can collect samples")
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=config.allow_additive_runtime_migration,
    )
    if not isinstance(
        restored.experiment, (ExperimentPlan, LayeredExperimentPlan),
    ):
        raise ValueError("factual collection requires a supported experiment plan")
    stream = FactualRequestStream(
        Path(config.request_stream_root) / config.checkpoint_branch,
        branch,
    )
    orphaned = stream.reconcile_through(restored.ref.logical_time)
    logical_time = restored.ref.logical_time + 1
    started_time = logical_time
    requests = 0
    events = 0
    transaction_id = (
        f"factual-{started_time}-{branch.head_checkpoint_id[:12]}"
    )
    staged_refs = []
    _sync(device)
    started = time.perf_counter()
    try:
        for _ in range(config.steps):
            tick = kernel.step(logical_time, restored.experiment)
            ref = stream.stage(
                transaction_id,
                tick,
                kernel.platform.projection.snapshot(),
                kernel.world.manifest(),
            )
            staged_refs.append(ref)
            requests += ref.requests
            events += ref.events
            logical_time += 1
        stream.commit_staged(transaction_id, tuple(staged_refs))
        learning_cursors = dict(restored.learning_cursors)
        learning_cursors["factual_request_stream"] = {
            "branch": branch.name,
            "stream_sha256": stream.stream_sha256,
            "first_collected_time": started_time,
            "last_collected_time": logical_time - 1,
            "partitions": len(stream.refs(training=True)),
        }
        checkpoint = store.save(
            kernel,
            logical_time - 1,
            restored.experiment,
            parent_checkpoint_id=branch.head_checkpoint_id,
            learning_cursors=learning_cursors,
        )
        updated = registry.advance(
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
        "schema": "factual-request-collection-review/v1",
        "quality_claim": "synthetic-world factual sample lineage only",
        "config": asdict(config),
        "branch": updated.name,
        "training_authority": updated.training_authority,
        "resumed_from_checkpoint": branch.head_checkpoint_id,
        "final_checkpoint_id": checkpoint.checkpoint_id,
        "logical_time": [started_time, logical_time - 1],
        "requests": requests,
        "events": events,
        "request_partitions": len(stream.refs(training=True)),
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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoint-branch", default="main")
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--ticks-per-day", type=int, default=8)
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument("--allow-additive-runtime-migration", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = collect_factual_requests(FactualCollectionConfig(
        checkpoint_root=args.checkpoint_root,
        request_stream_root=args.request_stream_root,
        steps=args.steps,
        checkpoint_branch=args.checkpoint_branch,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
        allow_code_migration=args.allow_code_migration,
        allow_additive_runtime_migration=args.allow_additive_runtime_migration,
    ))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
