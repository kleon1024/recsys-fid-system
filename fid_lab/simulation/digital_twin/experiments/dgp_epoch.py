"""Migrate the factual world to a new, explicitly versioned DGP epoch."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..engine import ExperimentPlan
from ..platform import CascadePolicy
from ..profile import STANDARD_FEED_PROFILE
from ..world.authority import (
    BehavioralSCMResponseAuthority,
    FormulaResponseAuthority,
)
from .retrieval_ladder import (
    RetrievalLadderConfig,
    _baseline_plan,
    _build_kernel,
)


DGP_EPOCH_CURSOR = "dgp_epoch_v5"
DGP_RUNTIME_CHANGES = {
    "world_manifest.response": (
        FormulaResponseAuthority.version,
        BehavioralSCMResponseAuthority.version,
    ),
}


@dataclass(frozen=True)
class DGPEpochMigrationConfig:
    checkpoint_root: str
    checkpoint_branch: str = "main"
    users: int = STANDARD_FEED_PROFILE.users
    items: int = STANDARD_FEED_PROFILE.items
    device: str = "cuda"
    seed: int = STANDARD_FEED_PROFILE.seed
    ticks_per_day: int = STANDARD_FEED_PROFILE.ticks_per_day


def _runtime_config(config: DGPEpochMigrationConfig) -> RetrievalLadderConfig:
    return RetrievalLadderConfig(
        checkpoint_root=config.checkpoint_root,
        checkpoint_branch=config.checkpoint_branch,
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
    )


def migrate_dgp_epoch(
    config: DGPEpochMigrationConfig,
) -> dict[str, object]:
    runtime = _runtime_config(config)
    _, kernel = _build_kernel(runtime)
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("DGP epoch migration requires the factual branch")
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=False,
        allow_additive_runtime_migration=True,
        approved_runtime_changes=DGP_RUNTIME_CHANGES,
    )
    if DGP_EPOCH_CURSOR in restored.learning_cursors:
        raise ValueError("DGP epoch v5 was already migrated")
    if not isinstance(restored.experiment, ExperimentPlan):
        raise ValueError("DGP migration requires a non-layered experiment")
    active = restored.experiment.policies[-1]
    if not isinstance(active, CascadePolicy):
        raise TypeError("DGP migration requires a cascade policy baseline")
    batches = kernel.event_log.partitions()
    kernel.world.rebuild_experience(batches)
    if not int(kernel.platform.projection.state.user_feed_exposure_cursor.sum()):
        kernel.platform.projection.rebuild_exposures(kernel.event_log.read())
    cursors = dict(restored.learning_cursors)
    pending = dict(cursors.get("feed_session_dedup_launch", {}))
    if pending and not pending.get("completed"):
        pending.update({
            "completed": True,
            "decision": "invalidated_dgp_v4",
            "validity": "excluded_from_v5_training_and_launch_decisions",
        })
        cursors["feed_session_dedup_launch"] = pending
    cursors[DGP_EPOCH_CURSOR] = {
        "schema": "dgp-epoch/v1",
        "effective_after_logical_time": restored.ref.logical_time,
        "previous_response_authority": FormulaResponseAuthority.version,
        "response_authority": BehavioralSCMResponseAuthority.version,
        "historical_request_stream_training_eligible": False,
        "migration_parent": branch.head_checkpoint_id,
    }
    plan = _baseline_plan(runtime, active, 9_100)
    checkpoint = store.save(
        kernel,
        restored.ref.logical_time,
        plan,
        parent_checkpoint_id=branch.head_checkpoint_id,
        learning_cursors=cursors,
    )
    registry.advance(
        branch.name,
        checkpoint.checkpoint_id,
        expected_head_checkpoint_id=branch.head_checkpoint_id,
    )
    return {
        "schema": "dgp-epoch-migration-review/v1",
        "quality_claim": "synthetic-world authority migration",
        "config": asdict(config),
        "logical_time": restored.ref.logical_time,
        "parent_checkpoint_id": branch.head_checkpoint_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "active_policy": active.name,
        "world_manifest": kernel.world.manifest(),
        "hidden_exposure_rows": int(
            kernel.world.users.exposure_cursor.sum()
        ),
        "users_with_exposure_memory": int(
            (kernel.world.users.exposure_cursor > 0).sum()
        ),
        "historical_samples_training_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-branch", default="main")
    parser.add_argument("--users", type=int, default=STANDARD_FEED_PROFILE.users)
    parser.add_argument("--items", type=int, default=STANDARD_FEED_PROFILE.items)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=STANDARD_FEED_PROFILE.seed)
    parser.add_argument(
        "--ticks-per-day", type=int,
        default=STANDARD_FEED_PROFILE.ticks_per_day,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = migrate_dgp_epoch(DGPEpochMigrationConfig(
        checkpoint_root=args.checkpoint_root,
        checkpoint_branch=args.checkpoint_branch,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
    ))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
