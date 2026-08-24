"""Sequential Feed retrieval launch reviews in one factual evolving world."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from ..catalog import build_public_catalog
from ..contracts import AppEventBatch, EventType, Surface
from ..engine import AtomicSimulationKernel, ExperimentPlan
from ..event_log import ObservableEventLog
from ..platform import (
    ROUTE_NAMES,
    CascadePolicy,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
)
from ..world import UserEcosystemWorld, UserWorldConfig


BASE_ROUTES = ("popular",)
ROUTE_LADDER = ("geo", "graph", "fresh", "long_tail", "ann")
COUNT_METRICS = {
    "play_3s": EventType.PLAY_3S,
    "long_view": EventType.LONG_VIEW,
    "complete": EventType.COMPLETE,
    "like": EventType.LIKE,
    "comment": EventType.COMMENT,
    "share": EventType.SHARE,
    "follow": EventType.FOLLOW,
    "negative": EventType.NEGATIVE,
    "session_end": EventType.SESSION_END,
}


@dataclass(frozen=True)
class RetrievalLadderConfig:
    users: int = 20_000
    items: int = 500_000
    burn_in_steps: int = 4
    experiment_steps: int = 8
    control_fraction: float = 0.20
    treatment_fraction: float = 0.20
    device: str = "cuda"
    seed: int = 809
    auto_promote: bool = True
    ticks_per_day: int = 8
    minimum_triggered_users: int = 500

    def __post_init__(self):
        if self.ticks_per_day <= 0 or self.minimum_triggered_users <= 1:
            raise ValueError("cadence and sample gate must be positive")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _user_metric(
    events: AppEventBatch,
    cell: int,
    users: int,
    event_type: EventType | None = None,
) -> torch.Tensor:
    selected = (events.experiment_cell == cell) & (events.user_id >= 0)
    if event_type is not None:
        selected &= events.event(event_type)
    result = torch.zeros(users, device=events.user_id.device)
    if event_type is EventType.DWELL:
        result.scatter_add_(
            0,
            events.user_id[selected],
            events.duration_ms[selected].float() / 1_000.0,
        )
    else:
        result.scatter_add_(
            0,
            events.user_id[selected],
            torch.ones(int(selected.sum()), device=events.user_id.device),
        )
    return result


def _cell_users(events: AppEventBatch, cell: int, users: int) -> torch.Tensor:
    impression = (events.experiment_cell == cell) & events.event(
        EventType.IMPRESSION
    )
    present = torch.zeros(users, device=events.user_id.device, dtype=torch.bool)
    present[events.user_id[impression]] = True
    return present


def _estimate(
    control: torch.Tensor,
    treatment: torch.Tensor,
) -> dict[str, float]:
    if len(control) < 2 or len(treatment) < 2:
        return {
            "control_mean": float("nan"),
            "treatment_mean": float("nan"),
            "absolute_delta": float("nan"),
            "relative_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    control_mean = control.mean()
    treatment_mean = treatment.mean()
    delta = treatment_mean - control_mean
    standard_error = torch.sqrt(
        control.var(unbiased=True) / max(len(control), 1)
        + treatment.var(unbiased=True) / max(len(treatment), 1)
    )
    return {
        "control_mean": float(control_mean),
        "treatment_mean": float(treatment_mean),
        "absolute_delta": float(delta),
        "relative_delta": float(delta / control_mean.clamp_min(1e-12)),
        "ci95_low": float(delta - 1.96 * standard_error),
        "ci95_high": float(delta + 1.96 * standard_error),
    }


def _analyze(
    batches: list[AppEventBatch], users: int,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    events = AppEventBatch.concatenate(tuple(batches))
    control_users = _cell_users(events, 0, users)
    treatment_users = _cell_users(events, 1, users)
    metrics = {}
    for name, event_type in {
        "dwell_seconds": EventType.DWELL,
        **COUNT_METRICS,
    }.items():
        values_control = _user_metric(events, 0, users, event_type)[control_users]
        values_treatment = _user_metric(
            events, 1, users, event_type,
        )[treatment_users]
        metrics[name] = _estimate(values_control, values_treatment)
    return metrics, {
        "control_triggered_users": int(control_users.sum()),
        "treatment_triggered_users": int(treatment_users.sum()),
    }


def _decision(
    metrics: dict[str, dict[str, float]],
    sample: dict[str, int],
    minimum_triggered_users: int,
) -> tuple[str, str]:
    if min(sample.values()) < minimum_triggered_users:
        return "hold", "triggered-user sample is below the preregistered gate"
    dwell = metrics["dwell_seconds"]
    negative = metrics["negative"]
    if not all(math.isfinite(value) for metric in metrics.values() for value in metric.values()):
        return "hold", "non-finite experiment metric"
    if dwell["ci95_high"] < 0.0:
        return "reject", "stay significantly decreases"
    if dwell["ci95_low"] <= 0.0:
        return "hold", "stay confidence interval crosses zero"
    if negative["ci95_low"] > 0.0:
        return "reject", "negative feedback significantly increases"
    return "promote", "stay improves and negative-feedback guardrail passes"


def _policy(name: str, version: int, routes: tuple[str, ...]) -> CascadePolicy:
    return CascadePolicy(
        name,
        coarse_version_id=1,
        fine_version_id=1,
        mix_version_id=1,
        recall_version_id=version,
        enabled_routes=routes,
    )


def _build_kernel(config: RetrievalLadderConfig):
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    catalog = build_public_catalog(
        items=config.items,
        creators=max(config.items // 20, 1),
        merchants=max(config.items // 100, 1),
        advertisers=max(config.items // 200, 1),
        topics=64,
        countries=12,
        regions_per_country=16,
        embedding_dim=32,
        platform_seed=config.seed + 1,
        device=device,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=config.users,
        topics=64,
        embedding_dim=32,
        countries=12,
        regions_per_country=16,
        environment_seed=config.seed + 2,
        ticks_per_day=config.ticks_per_day,
        future_signup_fraction=0.35,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=config.users, history_length=64),
        catalog,
        RetrievalConfig(
            route_k=24,
            merged_k=96,
            graph_neighbors=24,
            refresh_interval=1,
        ),
        RankingConfig(coarse_k=48, fine_k=16, expose_k=8),
    )
    return device, AtomicSimulationKernel(
        world,
        platform,
        ObservableEventLog(allowed_lateness=world.max_reporting_lag),
    )


def _run_review_window(
    kernel: AtomicSimulationKernel,
    plan: ExperimentPlan,
    logical_time: int,
    steps: int,
    route: str,
) -> tuple[
    list[AppEventBatch], dict[str, int], dict[str, int], dict[str, int], int,
]:
    batches = []
    requests = {"control": 0, "treatment": 0, "default": 0}
    route_hits = {"control": 0, "treatment": 0}
    stage_candidates = {
        "recall": 0, "coarse": 0, "fine": 0, "exposed": 0,
    }
    route_bit = 1 << ROUTE_NAMES.index(route)
    for _ in range(steps):
        tick = kernel.step(logical_time, plan)
        logical_time += 1
        batches.append(tick.response_events)
        for cell, name in ((0, "control"), (1, "treatment"), (-1, "default")):
            requests[name] += tick.cell_counts.get(cell, 0)
        trace = tick.candidate_trace
        if trace is None:
            continue
        for cell, name in ((0, "control"), (1, "treatment")):
            rows = trace.experiment_cell == cell
            route_hits[name] += int(
                (trace.recall_route_id[rows] & route_bit).any(dim=1).sum()
            )
        treatment = trace.experiment_cell == 1
        route_recall = (
            trace.recall_route_id[treatment] & route_bit
        ) > 0
        recall_item = trace.recall_item_id[treatment]
        stage_candidates["recall"] += int(route_recall.sum())
        for name, stage_item in (
            ("coarse", trace.coarse_item_id[treatment]),
            ("fine", trace.fine_item_id[treatment]),
            ("exposed", trace.exposed_item_id[treatment]),
        ):
            from_route = (
                (stage_item[:, :, None] == recall_item[:, None, :])
                & route_recall[:, None, :]
            ).any(dim=2)
            stage_candidates[name] += int(from_route.sum())
    return batches, requests, route_hits, stage_candidates, logical_time


def _run_one_review(
    kernel: AtomicSimulationKernel,
    config: RetrievalLadderConfig,
    active: CascadePolicy,
    active_routes: tuple[str, ...],
    route: str,
    index: int,
    logical_time: int,
) -> tuple[dict[str, object], CascadePolicy, tuple[str, ...], int]:
    proposed_routes = (*active_routes, route)
    treatment = _policy(
        f"feed-add-{route}-v{index + 1}", index + 1, proposed_routes,
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=treatment,
        experiment_seed=config.seed + 100 * index,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )
    batches, requests, route_hits, stage_candidates, logical_time = (
        _run_review_window(
        kernel, plan, logical_time, config.experiment_steps, route,
        )
    )
    metrics, sample = _analyze(batches, config.users)
    decision, reason = _decision(
        metrics, sample, config.minimum_triggered_users,
    )
    promoted = config.auto_promote and decision == "promote"
    review = {
        "launch_review": f"R-LR-{index:03d}",
        "changed_owner": "retrieval routes only",
        "control_routes": list(active_routes),
        "treatment_routes": list(proposed_routes),
        "added_route": route,
        "requests": requests,
        "route_request_hits": route_hits,
        "treatment_route_stage_candidates": stage_candidates,
        "treatment_route_pass_rate": {
            stage: count / max(stage_candidates["recall"], 1)
            for stage, count in stage_candidates.items()
        },
        "sample": sample,
        "metrics_per_triggered_user": metrics,
        "decision": decision,
        "reason": reason,
        "promoted_to_next_baseline": promoted,
    }
    return (
        review,
        treatment if promoted else active,
        proposed_routes if promoted else active_routes,
        logical_time,
    )


def run_retrieval_ladder(config: RetrievalLadderConfig) -> dict[str, object]:
    device, kernel = _build_kernel(config)
    active_routes = BASE_ROUTES
    active = _policy("feed-popular-v1", 1, active_routes)
    logical_time = 0
    _sync(device)
    started = time.perf_counter()
    burn_in = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=active,
        experiment_seed=config.seed + 10,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )
    for _ in range(config.burn_in_steps):
        kernel.step(logical_time, burn_in)
        logical_time += 1
    reviews = []
    for index, route in enumerate(ROUTE_LADDER, start=1):
        review, active, active_routes, logical_time = _run_one_review(
            kernel, config, active, active_routes, route, index, logical_time,
        )
        reviews.append(review)
    _sync(device)
    elapsed = time.perf_counter() - started
    return {
        "scope": "v4-feed-sequential-retrieval-launch-reviews",
        "quality_claim": "synthetic-world causal evidence only",
        "config": asdict(config),
        "invariant": (
            "one factual slate per request; hidden world consumes only served "
            "events; coarse/fine/mix remain fixed"
        ),
        "metric_definition": (
            "clustered intent-to-treat means per triggered user over each "
            "experiment window; LT exchange is deliberately not invented"
        ),
        "reviews": reviews,
        "final_active_routes": list(active_routes),
        "elapsed_seconds": elapsed,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--burn-in-steps", type=int, default=4)
    parser.add_argument("--experiment-steps", type=int, default=8)
    parser.add_argument("--control-fraction", type=float, default=0.20)
    parser.add_argument("--treatment-fraction", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--ticks-per-day", type=int, default=8)
    parser.add_argument("--minimum-triggered-users", type=int, default=500)
    parser.add_argument("--no-auto-promote", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_retrieval_ladder(RetrievalLadderConfig(
        users=args.users,
        items=args.items,
        burn_in_steps=args.burn_in_steps,
        experiment_steps=args.experiment_steps,
        control_fraction=args.control_fraction,
        treatment_fraction=args.treatment_fraction,
        device=args.device,
        seed=args.seed,
        auto_promote=not args.no_auto_promote,
        ticks_per_day=args.ticks_per_day,
        minimum_triggered_users=args.minimum_triggered_users,
    ))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
