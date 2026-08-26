"""Launch a persistent, propensity-logged item cold-start exploration layer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..contracts import ContentKind, SelectionPolicyKind, Surface
from ..engine import ExperimentPlan
from ..learning.request_stream import FactualRequestStream
from ..platform import CascadePolicy, ContentLifecycle
from ..profile import STANDARD_FEED_PROFILE
from .launch_review.metrics import analyze_experiment
from .layered import LayeredExperimentPlan, PolicyLayer
from .retrieval_ladder import (
    RetrievalLadderConfig,
    _baseline_plan,
    _build_kernel,
    _sync,
)


@dataclass(frozen=True)
class ColdStartLaunchConfig:
    checkpoint_root: str
    request_stream_root: str
    checkpoint_branch: str = "main"
    users: int = STANDARD_FEED_PROFILE.users
    items: int = STANDARD_FEED_PROFILE.items
    device: str = "cuda"
    seed: int = STANDARD_FEED_PROFILE.seed
    ticks_per_day: int = STANDARD_FEED_PROFILE.ticks_per_day
    experiment_steps: int = 8
    control_fraction: float = 0.2
    treatment_fraction: float = 0.2
    within_treatment_rate: float = 0.10
    minimum_triggered_users: int = 500
    minimum_cold_exposures: int = 30
    maximum_attempts: int = 5
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False
    launch_id: str = "R-LR-012"
    cursor_key: str = "cold_start_launch_v2"


def _runtime_config(config: ColdStartLaunchConfig) -> RetrievalLadderConfig:
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


def _active_and_plan(
    restored,
    config: ColdStartLaunchConfig,
) -> tuple[CascadePolicy, LayeredExperimentPlan]:
    if isinstance(restored.experiment, ExperimentPlan):
        active = restored.experiment.policies[-1]
    elif isinstance(restored.experiment, LayeredExperimentPlan):
        active = restored.experiment.base_policy
    else:
        raise TypeError("unsupported experiment plan")
    routes = tuple(dict.fromkeys((*active.enabled_routes, "cold_start")))
    plan = LayeredExperimentPlan(
        active,
        (PolicyLayer(
            name="item-cold-start-exploration",
            salt=config.seed + 11_000,
            changes={
                "recall_version_id": max(active.recall_version_id + 1, 11),
                "enabled_routes": routes,
                "cold_start_exploration_rate": config.within_treatment_rate,
            },
            control_fraction=config.control_fraction,
            treatment_fraction=config.treatment_fraction,
            eligible_surfaces=(int(Surface.FEED),),
        ),),
    )
    return active, plan


def _window_counts(trace, catalog) -> dict[str, int | float]:
    feed = trace.surface == int(Surface.FEED)
    treatment = feed & (trace.experiment_cell == 2)
    randomized = treatment & (
        trace.selection_policy_kind == int(SelectionPolicyKind.RANDOMIZED)
    )
    cold_bit = 1 << trace.manifest.route_names.index("cold_start")
    safe = trace.recall_item_id.clamp_min(0)
    cold_candidate = (
        (trace.recall_item_id >= 0)
        & ((trace.recall_route_id & cold_bit) > 0)
        & (trace.recall_lifecycle_id == int(ContentLifecycle.COLD_START))
        & (catalog.content_kind[safe] == int(ContentKind.SHORT_VIDEO))
    )
    already_exposed = (
        trace.recall_item_id[:, :, None] == trace.exposed_item_id[:, None, :]
    ).any(dim=2)
    supported = treatment & (
        randomized | (cold_candidate & ~already_exposed).any(dim=1)
    )
    last = trace.exposed_item_id[:, -1]
    last_match = last[:, None] == trace.recall_item_id
    last_cold = (
        last_match
        & cold_candidate
        & (last[:, None] >= 0)
    ).any(dim=1)
    probability = trace.exposure_probability[randomized, -1]
    return {
        "feed_requests": int(feed.sum()),
        "control_requests": int((feed & (trace.experiment_cell == 1)).sum()),
        "treatment_requests": int(treatment.sum()),
        "supported_treatment_requests": int(supported.sum()),
        "randomized_requests": int(randomized.sum()),
        "cold_start_exposures": int((randomized & last_cold).sum()),
        "invalid_cold_exposures": int((randomized & ~last_cold).sum()),
        "minimum_logged_propensity": (
            float(probability.min()) if len(probability) else 1.0
        ),
        "maximum_logged_propensity": (
            float(probability.max()) if len(probability) else 0.0
        ),
    }


def _merge_counts(
    current: dict[str, int | float],
    previous: Mapping[str, int | float] | None,
) -> dict[str, int | float]:
    if previous is None:
        return current
    result = {}
    for key, value in current.items():
        if key == "minimum_logged_propensity":
            result[key] = min(float(value), float(previous[key]))
        elif key == "maximum_logged_propensity":
            result[key] = max(float(value), float(previous[key]))
        else:
            result[key] = int(value) + int(previous[key])
    return result


def _rates(counts: Mapping[str, int | float]) -> dict[str, float]:
    return {
        "global_randomized_request_rate": int(counts["randomized_requests"])
        / max(int(counts["feed_requests"]), 1),
        "treatment_randomized_request_rate": int(counts["randomized_requests"])
        / max(int(counts["treatment_requests"]), 1),
        "treatment_candidate_support_rate": int(
            counts["supported_treatment_requests"]
        ) / max(int(counts["treatment_requests"]), 1),
    }


def _decision(
    metrics: Mapping[str, Mapping[str, float]],
    sample: Mapping[str, int],
    counts: Mapping[str, int | float],
    rates: Mapping[str, float],
    config: ColdStartLaunchConfig,
) -> tuple[str, str]:
    if min(sample.values()) < config.minimum_triggered_users:
        return "hold", "triggered-user sample is below the preregistered gate"
    if int(counts["cold_start_exposures"]) < config.minimum_cold_exposures:
        return "hold", "cold-start exposure sample is below the gate"
    if int(counts["invalid_cold_exposures"]):
        return "reject", "randomized slot violated cold-start provenance"
    if not 0.01 <= rates["global_randomized_request_rate"] <= 0.03:
        return "reject", "global randomized traffic differs from the 2% budget"
    if not 0.07 <= rates["treatment_randomized_request_rate"] <= 0.13:
        return "reject", "within-treatment randomization differs from 10%"
    if rates["treatment_candidate_support_rate"] < 0.80:
        return "reject", "cold-start route support is below 80%"
    if float(counts["minimum_logged_propensity"]) <= 0.0:
        return "reject", "randomized exposure has zero propensity"
    if not all(
        math.isfinite(value)
        for metric in metrics.values() for value in metric.values()
    ):
        return "hold", "non-finite experiment metric"
    dwell = metrics["dwell_seconds"]
    noninferiority = -0.05 * dwell["control_mean"]
    if dwell["ci95_high"] < noninferiority:
        return "reject", "stay violates the 5% exploration noninferiority margin"
    if dwell["ci95_low"] < noninferiority:
        return "hold", "stay noninferiority is not yet powered"
    if metrics["negative"]["ci95_low"] > 0.0:
        return "reject", "negative feedback significantly increases"
    return "accept_layer", "traffic, propensity and user guardrails pass"


def _run_window(
    kernel, stream, plan, logical_time, config, transaction_id,
):
    counts = None
    staged_refs = []
    for _ in range(config.experiment_steps):
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
        counts = _merge_counts(
            _window_counts(tick.candidate_trace, kernel.world.catalog), counts,
        )
        logical_time += 1
    if counts is None:
        raise ValueError("cold-start window produced no ticks")
    return logical_time, counts, tuple(staged_refs)


def run_cold_start_launch(
    config: ColdStartLaunchConfig,
) -> dict[str, object]:
    device, kernel = _build_kernel(_runtime_config(config))
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    branch = registry.get(config.checkpoint_branch)
    restored = store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=(
            config.allow_additive_runtime_migration
        ),
    )
    cursor = dict(restored.learning_cursors.get(config.cursor_key, {}))
    if cursor.get("completed"):
        raise ValueError("cold-start launch is already complete")
    active, plan = _active_and_plan(restored, config)
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
        logical_time, counts, staged_refs = _run_window(
            kernel, stream, plan, logical_time, config, transaction_id,
        )
        stream.commit_staged(transaction_id, staged_refs)
    except Exception:
        stream.abort_staged(transaction_id)
        stream.reconcile_through(restored.ref.logical_time)
        raise
    counts = _merge_counts(counts, cursor.get("counts"))
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    events = events.select(events.ingest_time >= start_time)
    metrics, sample = analyze_experiment(
        events, config.users, control_cell=1, treatment_cell=2,
    )
    rates = _rates(counts)
    decision, reason = _decision(metrics, sample, counts, rates, config)
    if decision == "hold" and attempt >= config.maximum_attempts:
        decision, reason = "stop_inconclusive", "maximum review windows reached"
    completed = decision in {"accept_layer", "reject", "stop_inconclusive"}
    review = {
        "launch_review": config.launch_id,
        "attempt": attempt,
        "analysis_window": [start_time, logical_time - 1],
        "sample": sample,
        "traffic_counts": counts,
        "traffic_rates": rates,
        "metrics_per_triggered_user": metrics,
        "decision": decision,
        "reason": reason,
    }
    reviews = [*cursor.get("reviews", []), review]
    cursors = dict(restored.learning_cursors)
    cursors[config.cursor_key] = {
        "attempt": attempt,
        "analysis_start_time": start_time,
        "counts": counts,
        "reviews": reviews,
        "completed": completed,
        "decision": decision,
    }
    cursors["factual_request_stream"] = {
        "branch": branch.name,
        "stream_sha256": stream.stream_sha256,
        "last_collected_time": logical_time - 1,
        "partitions": len(stream.refs(training=True)),
    }
    checkpoint_plan = (
        plan if decision in {"accept_layer", "hold"}
        else _baseline_plan(_runtime_config(config), active, 11_100)
    )
    try:
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
        stream.reconcile_through(restored.ref.logical_time)
        raise
    _sync(device)
    return {
        "schema": "cold-start-launch-review/v1",
        "quality_claim": "synthetic-world causal evidence only",
        "config": asdict(config),
        "resumed_from_checkpoint": branch.head_checkpoint_id,
        "final_checkpoint_id": checkpoint.checkpoint_id,
        "active_base_policy": active.name,
        "persistent_layer": decision == "accept_layer",
        "review": review,
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
    parser.add_argument("--users", type=int, default=STANDARD_FEED_PROFILE.users)
    parser.add_argument("--items", type=int, default=STANDARD_FEED_PROFILE.items)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=STANDARD_FEED_PROFILE.seed)
    parser.add_argument(
        "--ticks-per-day", type=int,
        default=STANDARD_FEED_PROFILE.ticks_per_day,
    )
    parser.add_argument("--experiment-steps", type=int, default=8)
    parser.add_argument("--within-treatment-rate", type=float, default=0.10)
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument(
        "--allow-additive-runtime-migration", action="store_true",
    )
    parser.add_argument("--launch-id", default="R-LR-012")
    parser.add_argument("--cursor-key", default="cold_start_launch_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_cold_start_launch(ColdStartLaunchConfig(
        checkpoint_root=args.checkpoint_root,
        request_stream_root=args.request_stream_root,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
        experiment_steps=args.experiment_steps,
        within_treatment_rate=args.within_treatment_rate,
        allow_code_migration=args.allow_code_migration,
        allow_additive_runtime_migration=(
            args.allow_additive_runtime_migration
        ),
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
