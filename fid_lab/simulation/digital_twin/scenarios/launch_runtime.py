"""Transactional factual-world runtime shared by business Launch Reviews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..experiments.retrieval_ladder import RetrievalLadderConfig, _build_kernel
from ..learning.request_stream import FactualRequestStream


@dataclass(frozen=True)
class FactualLaunchRuntime:
    device: torch.device
    kernel: object
    store: WorldCheckpointStore
    registry: WorldBranchRegistry
    branch: object
    restored: object
    stream: FactualRequestStream
    reconciled_orphans: tuple[object, ...]
    repaired_history_violations: int


def open_factual_launch(config) -> FactualLaunchRuntime:
    runtime_config = RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
        experiment_steps=config.experiment_steps,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
    )
    device, kernel = _build_kernel(runtime_config)
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("Launch Review requires the factual branch")
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=(
            config.allow_additive_runtime_migration
        ),
    )
    history_violations = (
        kernel.platform.projection.history_chronology_violations()
    )
    if history_violations:
        kernel.platform.projection.rebuild_history(
            kernel.event_log.partitions()
        )
    stream = FactualRequestStream(
        Path(config.request_stream_root) / branch.name, branch,
    )
    orphans = stream.reconcile_through(restored.ref.logical_time)
    return FactualLaunchRuntime(
        device,
        kernel,
        store,
        registry,
        branch,
        restored,
        stream,
        orphans,
        history_violations,
    )


def publish_factual_launch(
    runtime: FactualLaunchRuntime,
    *,
    transaction_id: str,
    staged_refs: tuple[object, ...],
    logical_time: int,
    plan,
    learning_cursors,
):
    try:
        runtime.stream.commit_staged(transaction_id, staged_refs)
        checkpoint = runtime.store.save(
            runtime.kernel,
            logical_time,
            plan,
            parent_checkpoint_id=runtime.branch.head_checkpoint_id,
            learning_cursors=learning_cursors,
        )
        runtime.registry.advance(
            runtime.branch.name,
            checkpoint.checkpoint_id,
            expected_head_checkpoint_id=runtime.branch.head_checkpoint_id,
        )
        return checkpoint
    except Exception:
        runtime.stream.abort_staged(transaction_id)
        runtime.stream.reconcile_through(runtime.restored.ref.logical_time)
        raise
