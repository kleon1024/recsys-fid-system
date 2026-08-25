"""Repair route ownership so Feed experiments cannot disable business surfaces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..engine import ExperimentPlan
from ..platform import BUSINESS_ROUTE_NAMES, CascadePolicy
from .layered import LayeredExperimentPlan
from .retrieval_ladder import (
    RetrievalLadderConfig,
    _baseline_plan,
    _build_kernel,
)


@dataclass(frozen=True)
class SurfaceRouteRecoveryConfig:
    checkpoint_root: str
    checkpoint_branch: str = "main"
    users: int = 20_000
    items: int = 500_000
    device: str = "cuda"
    seed: int = 809
    ticks_per_day: int = 8


def _runtime_config(config: SurfaceRouteRecoveryConfig) -> RetrievalLadderConfig:
    return RetrievalLadderConfig(
        checkpoint_root=config.checkpoint_root,
        checkpoint_branch=config.checkpoint_branch,
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
    )


def _base_policy(plan) -> CascadePolicy:
    if isinstance(plan, ExperimentPlan):
        return plan.policies[-1]
    if isinstance(plan, LayeredExperimentPlan):
        return plan.base_policy
    raise TypeError("surface recovery requires a supported experiment plan")


def recover_surface_routes(
    config: SurfaceRouteRecoveryConfig,
) -> dict[str, object]:
    runtime = _runtime_config(config)
    _, kernel = _build_kernel(runtime)
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("surface recovery requires the factual branch")
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=False,
    )
    if "surface_route_recovery" in restored.learning_cursors:
        raise ValueError("surface route ownership was already recovered")
    previous = _base_policy(restored.experiment)
    active = replace(
        previous,
        name="multi-surface-route-isolated-v1",
        recall_version_id=max(previous.recall_version_id + 1, 12),
        enabled_business_routes=BUSINESS_ROUTE_NAMES,
        cold_start_exploration_rate=0.0,
    )
    cursors = dict(restored.learning_cursors)
    for key in tuple(cursors):
        if not key.startswith("cold_start_launch"):
            continue
        cold = dict(cursors[key])
        if cold and not cold.get("completed"):
            cold.update({
                "completed": True,
                "decision": "invalidated_missing_business_route_authority",
            })
            cursors[key] = cold
    cursors["surface_route_recovery"] = {
        "schema": "surface-route-recovery/v1",
        "logical_time": restored.ref.logical_time,
        "previous_policy": previous.name,
        "active_policy": active.name,
        "fixed_business_routes": list(BUSINESS_ROUTE_NAMES),
        "invalidated_launch": "R-LR-011",
        "migration_parent": branch.head_checkpoint_id,
    }
    plan = _baseline_plan(runtime, active, 12_100)
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
        "schema": "surface-route-recovery-review/v1",
        "quality_claim": "synthetic serving invariant repair",
        "config": asdict(config),
        "logical_time": restored.ref.logical_time,
        "parent_checkpoint_id": branch.head_checkpoint_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "previous_policy": previous.name,
        "active_policy": active.name,
        "feed_routes": list(active.enabled_routes),
        "business_routes": list(active.enabled_business_routes),
        "invalidated_launch": "R-LR-011",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-branch", default="main")
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--ticks-per-day", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = recover_surface_routes(SurfaceRouteRecoveryConfig(
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
