"""Sequential Feed retrieval launch reviews in one factual evolving world."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time
from typing import Literal, Mapping

import torch

from ..catalog import build_public_catalog
from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
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
from ..profile import STANDARD_FEED_PROFILE, SimulationProfile
from ..world import UserEcosystemWorld, UserWorldConfig
from ..world.authority import (
    FactualResponseArtifact,
    FormulaResponseAuthority,
    load_factual_response_authority,
)


BASE_ROUTES = ("random",)
ROUTE_LADDER = (
    "popular",
    "cold_start",
    "recent_ann",
    "recent_graph",
    "following",
    "hot",
    "evergreen",
)
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
    users: int = STANDARD_FEED_PROFILE.users
    items: int = STANDARD_FEED_PROFILE.items
    burn_in_steps: int = 4
    experiment_steps: int = 8
    control_fraction: float = 0.20
    treatment_fraction: float = 0.20
    device: str = "cuda"
    seed: int = STANDARD_FEED_PROFILE.seed
    auto_promote: bool = True
    ticks_per_day: int = STANDARD_FEED_PROFILE.ticks_per_day
    minimum_triggered_users: int = 500
    checkpoint_root: str | None = None
    checkpoint_branch: str = "main"
    resume_checkpoint_id: str | None = None
    max_reviews: int | None = None
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False
    max_attempts_per_review: int = 3
    response_authority_mode: Literal["formula_oracle", "neural_feed"] = (
        "formula_oracle"
    )
    response_artifact_dir: str | None = None
    response_manifest_sha256: str | None = None
    response_member_index: int = 0
    maximum_support_fallback_rate: float = 0.03
    event_log_root: str | None = None

    def __post_init__(self):
        if self.ticks_per_day <= 0 or self.minimum_triggered_users <= 1:
            raise ValueError("cadence and sample gate must be positive")
        if self.max_reviews is not None and self.max_reviews <= 0:
            raise ValueError("max_reviews must be positive when provided")
        if self.max_attempts_per_review <= 0:
            raise ValueError("max_attempts_per_review must be positive")
        if not self.checkpoint_branch:
            raise ValueError("checkpoint_branch must not be empty")
        if self.response_authority_mode not in {"formula_oracle", "neural_feed"}:
            raise ValueError("response authority mode is unsupported")
        if self.response_authority_mode == "neural_feed" and (
            not self.response_artifact_dir or not self.response_manifest_sha256
        ):
            raise ValueError("NeuralSCM factual runtime requires an artifact ref")

    @property
    def simulation_profile(self) -> SimulationProfile:
        return replace(
            STANDARD_FEED_PROFILE,
            users=self.users,
            items=self.items,
            ticks_per_day=self.ticks_per_day,
            seed=self.seed,
        )


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
    batches: list[AppEventBatch],
    users: int,
    control_cell: int = 0,
    treatment_cell: int = 1,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    events = AppEventBatch.concatenate(tuple(batches))
    control_users = _cell_users(events, control_cell, users)
    treatment_users = _cell_users(events, treatment_cell, users)
    metrics = {}
    for name, event_type in {
        "dwell_seconds": EventType.DWELL,
        **COUNT_METRICS,
    }.items():
        values_control = _user_metric(
            events, control_cell, users, event_type,
        )[control_users]
        values_treatment = _user_metric(
            events, treatment_cell, users, event_type,
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


def _policy(
    name: str,
    version: int,
    routes: tuple[str, ...],
    ticks_per_day: int,
) -> CascadePolicy:
    return CascadePolicy(
        name,
        coarse_version_id=0,
        fine_version_id=0,
        mix_version_id=1,
        recall_version_id=version,
        enabled_routes=routes,
        feed_exposure_dedup_ticks=30 * ticks_per_day,
        feed_session_dedup=True,
    )


def _baseline_plan(
    config: RetrievalLadderConfig, active: CascadePolicy, seed_offset: int,
) -> ExperimentPlan:
    return ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=active,
        experiment_seed=config.seed + seed_offset,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )


def _build_kernel(config: RetrievalLadderConfig):
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profile = config.simulation_profile
    catalog = build_public_catalog(
        items=profile.items,
        creators=max(profile.items // 20, 1),
        merchants=max(profile.items // 100, 1),
        advertisers=max(profile.items // 200, 1),
        topics=profile.topics,
        countries=profile.countries,
        regions_per_country=profile.regions_per_country,
        embedding_dim=profile.embedding_dim,
        platform_seed=config.seed + 1,
        device=device,
    )
    response_authority = (
        FormulaResponseAuthority()
        if config.response_authority_mode == "formula_oracle"
        else load_factual_response_authority(
            FactualResponseArtifact(
                artifact_dir=str(config.response_artifact_dir),
                manifest_sha256=str(config.response_manifest_sha256),
                member_index=config.response_member_index,
                maximum_support_fallback_rate=(
                    config.maximum_support_fallback_rate
                ),
            ),
            device,
        )
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=profile.users,
        topics=profile.topics,
        embedding_dim=profile.embedding_dim,
        countries=profile.countries,
        regions_per_country=profile.regions_per_country,
        environment_seed=config.seed + 2,
        ticks_per_day=profile.ticks_per_day,
        future_signup_fraction=0.35,
    ), catalog, response_authority=response_authority)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(
            users=profile.users,
            history_length=profile.history_length,
            feed_exposure_history_length=profile.feed_exposure_history_length,
            ticks_per_day=profile.ticks_per_day,
        ),
        catalog,
        RetrievalConfig(
            route_k=profile.route_k,
            merged_k=profile.merged_k,
            graph_neighbors=24,
            refresh_interval=1,
        ),
        RankingConfig(
            coarse_k=profile.coarse_k,
            fine_k=profile.fine_k,
            expose_k=profile.expose_k,
        ),
    )
    event_log = (
        ObservableEventLog(allowed_lateness=world.max_reporting_lag)
        if config.event_log_root is None else ObservableEventLog(
            allowed_lateness=world.max_reporting_lag,
            root=Path(config.event_log_root),
        )
    )
    return device, AtomicSimulationKernel(world, platform, event_log)


def _run_review_window(
    kernel: AtomicSimulationKernel,
    plan: ExperimentPlan,
    logical_time: int,
    steps: int,
    route: str,
) -> tuple[
    dict[str, int], dict[str, int], dict[str, int], int,
]:
    requests = {"control": 0, "treatment": 0, "default": 0}
    route_hits = {"control": 0, "treatment": 0}
    stage_candidates = {
        "recall": 0, "coarse": 0, "fine": 0, "exposed": 0,
    }
    route_bit = 1 << ROUTE_NAMES.index(route)
    for _ in range(steps):
        tick = kernel.step(logical_time, plan)
        logical_time += 1
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
    return requests, route_hits, stage_candidates, logical_time


def _add_counts(
    current: dict[str, int], prior: Mapping[str, int] | None,
) -> dict[str, int]:
    if prior is None:
        return current
    if set(current) != set(prior):
        raise ValueError("pending launch counters changed schema")
    return {name: current[name] + int(prior[name]) for name in current}


def _pending_review(
    route: str,
    route_index: int,
    start_time: int,
    attempt: int,
    requests: dict[str, int],
    route_hits: dict[str, int],
    stage_candidates: dict[str, int],
) -> dict[str, object]:
    return {
        "route": route,
        "route_index": route_index,
        "start_time": start_time,
        "attempt": attempt,
        "requests": requests,
        "route_hits": route_hits,
        "stage_candidates": stage_candidates,
    }


def _run_one_review(
    kernel: AtomicSimulationKernel,
    config: RetrievalLadderConfig,
    active: CascadePolicy,
    active_routes: tuple[str, ...],
    route: str,
    index: int,
    logical_time: int,
    pending: Mapping[str, object] | None = None,
) -> tuple[
    dict[str, object],
    CascadePolicy,
    tuple[str, ...],
    int,
    dict[str, object] | None,
    ExperimentPlan,
]:
    proposed_routes = (*active_routes, route)
    treatment = _policy(
        f"feed-add-{route}-v{index + 1}", index + 1, proposed_routes,
        config.ticks_per_day,
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=treatment,
        experiment_seed=config.seed + 100 * index,
        control_fraction=config.control_fraction,
        treatment_fraction=config.treatment_fraction,
        eligible_surfaces=(int(Surface.FEED),),
    )
    start_time = logical_time if pending is None else int(pending["start_time"])
    attempt = 1 if pending is None else int(pending["attempt"]) + 1
    if pending is not None and (
        pending["route"] != route or int(pending["route_index"]) != index - 1
    ):
        raise ValueError("pending launch cursor points to a different route")
    requests, route_hits, stage_candidates, logical_time = (
        _run_review_window(
            kernel, plan, logical_time, config.experiment_steps, route,
        )
    )
    requests = _add_counts(
        requests, None if pending is None else pending["requests"],
    )
    route_hits = _add_counts(
        route_hits, None if pending is None else pending["route_hits"],
    )
    stage_candidates = _add_counts(
        stage_candidates,
        None if pending is None else pending["stage_candidates"],
    )
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    events = events.select(events.ingest_time >= start_time)
    metrics, sample = _analyze([events], config.users)
    decision, reason = _decision(
        metrics, sample, config.minimum_triggered_users,
    )
    if decision == "hold" and attempt >= config.max_attempts_per_review:
        decision = "stop_inconclusive"
        reason = "maximum review windows reached without conclusive lift"
    promoted = config.auto_promote and decision == "promote"
    review = {
        "launch_review": f"R-LR-{index:03d}",
        "attempt": attempt,
        "analysis_start_time": start_time,
        "analysis_end_time": logical_time - 1,
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
    next_pending = (
        _pending_review(
            route,
            index - 1,
            start_time,
            attempt,
            requests,
            route_hits,
            stage_candidates,
        )
        if decision == "hold" else None
    )
    return (
        review,
        treatment if promoted else active,
        proposed_routes if promoted else active_routes,
        logical_time,
        next_pending,
        plan,
    )


def _restore_or_burn_in(
    config: RetrievalLadderConfig,
    kernel: AtomicSimulationKernel,
    store: WorldCheckpointStore | None,
    registry: WorldBranchRegistry | None,
    resume_checkpoint_id: str | None,
) -> tuple[
    CascadePolicy,
    tuple[str, ...],
    int,
    list[dict[str, object]],
    list[str],
    int,
    dict[str, object] | None,
    str,
]:
    active_routes = BASE_ROUTES
    active = _policy(
        "feed-random-v1", 1, active_routes, config.ticks_per_day,
    )
    if resume_checkpoint_id is not None:
        if store is None:
            raise ValueError("resume requires checkpoint_root")
        restored = store.restore(
            kernel,
            resume_checkpoint_id,
            require_code_match=not config.allow_code_migration,
            allow_additive_runtime_migration=(
                config.allow_additive_runtime_migration
            ),
        )
        cursor = restored.learning_cursors.get("retrieval_ladder")
        if not isinstance(cursor, dict):
            raise ValueError("checkpoint has no retrieval ladder cursor")
        if not isinstance(restored.experiment, ExperimentPlan):
            raise ValueError("retrieval ladder checkpoint has a layered plan")
        return (
            restored.experiment.policies[-1],
            tuple(cursor["active_routes"]),
            restored.ref.logical_time + 1,
            list(cursor["reviews"]),
            [restored.ref.checkpoint_id],
            int(cursor["next_route_index"]),
            cursor.get("pending_review"),
            restored.ref.checkpoint_id,
        )
    logical_time = 0
    burn_in = _baseline_plan(config, active, 10)
    for _ in range(config.burn_in_steps):
        kernel.step(logical_time, burn_in)
        logical_time += 1
    if store is None:
        return active, active_routes, logical_time, [], [], 0, None, ""
    ref = store.save(
        kernel,
        logical_time - 1,
        burn_in,
        learning_cursors={
            "retrieval_ladder": {
                "next_route_index": 0,
                "active_routes": list(active_routes),
                "reviews": [],
                "pending_review": None,
            },
        },
    )
    if registry is not None:
        if config.checkpoint_branch != "main":
            raise ValueError("a new factual world must start on the main branch")
        registry.initialize_main(ref.checkpoint_id)
    return (
        active,
        active_routes,
        logical_time,
        [],
        [ref.checkpoint_id],
        0,
        None,
        ref.checkpoint_id,
    )


def _checkpoint_control(
    config: RetrievalLadderConfig,
) -> tuple[
    WorldCheckpointStore | None,
    WorldBranchRegistry | None,
    str | None,
]:
    if config.checkpoint_root is None:
        return None, None, config.resume_checkpoint_id
    store = WorldCheckpointStore(Path(config.checkpoint_root))
    registry = WorldBranchRegistry(store)
    resume_checkpoint_id = config.resume_checkpoint_id
    try:
        branch = registry.get(config.checkpoint_branch)
    except KeyError:
        if resume_checkpoint_id is None:
            return store, registry, None
        if config.checkpoint_branch != "main":
            raise ValueError(
                "create a diagnostic branch before resuming it",
            ) from None
        registry.initialize_main(resume_checkpoint_id)
        return store, registry, resume_checkpoint_id
    if (
        resume_checkpoint_id is not None
        and resume_checkpoint_id != branch.head_checkpoint_id
    ):
        raise ValueError(
            "resume checkpoint differs from registered branch head",
        )
    return store, registry, branch.head_checkpoint_id


def _save_review_checkpoint(
    store: WorldCheckpointStore,
    registry: WorldBranchRegistry | None,
    config: RetrievalLadderConfig,
    kernel: AtomicSimulationKernel,
    plan: ExperimentPlan,
    active: CascadePolicy,
    active_routes: tuple[str, ...],
    reviews: list[dict[str, object]],
    pending_review: dict[str, object] | None,
    next_route_index: int,
    logical_time: int,
    index: int,
    parent_checkpoint_id: str,
) -> str:
    checkpoint_plan = (
        plan
        if pending_review is not None
        else _baseline_plan(config, active, 10_000 + index)
    )
    ref = store.save(
        kernel,
        logical_time - 1,
        checkpoint_plan,
        parent_checkpoint_id=parent_checkpoint_id,
        learning_cursors={
            "retrieval_ladder": {
                "next_route_index": next_route_index,
                "active_routes": list(active_routes),
                "reviews": reviews,
                "pending_review": pending_review,
            },
        },
    )
    if registry is not None:
        registry.advance(
            config.checkpoint_branch,
            ref.checkpoint_id,
            expected_head_checkpoint_id=parent_checkpoint_id,
        )
    return ref.checkpoint_id


def run_retrieval_ladder(config: RetrievalLadderConfig) -> dict[str, object]:
    device, kernel = _build_kernel(config)
    store, registry, resume_checkpoint_id = _checkpoint_control(config)
    _sync(device)
    started = time.perf_counter()
    (
        active,
        active_routes,
        logical_time,
        reviews,
        checkpoint_ids,
        next_route_index,
        pending_review,
        parent_checkpoint_id,
    ) = _restore_or_burn_in(
        config,
        kernel,
        store,
        registry,
        resume_checkpoint_id,
    )
    review_windows = 0
    while next_route_index < len(ROUTE_LADDER):
        if config.max_reviews is not None and review_windows >= config.max_reviews:
            break
        route_offset = next_route_index
        route = ROUTE_LADDER[route_offset]
        index = route_offset + 1
        review, active, active_routes, logical_time, pending_review, plan = (
            _run_one_review(
                kernel,
                config,
                active,
                active_routes,
                route,
                index,
                logical_time,
                pending_review,
            )
        )
        reviews.append(review)
        review_windows += 1
        if pending_review is None:
            next_route_index += 1
        if store is not None:
            parent_checkpoint_id = _save_review_checkpoint(
                store,
                registry,
                config,
                kernel,
                plan,
                active,
                active_routes,
                reviews,
                pending_review,
                next_route_index,
                logical_time,
                index,
                parent_checkpoint_id,
            )
            checkpoint_ids.append(parent_checkpoint_id)
        if pending_review is not None:
            break
    _sync(device)
    elapsed = time.perf_counter() - started
    return {
        "scope": "v4-feed-sequential-retrieval-launch-reviews",
        "quality_claim": "synthetic-world causal evidence only",
        "config": asdict(config),
        "simulation_profile": config.simulation_profile.manifest(),
        "simulation_profile_hash": config.simulation_profile.profile_hash,
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
        "checkpoint_ids": checkpoint_ids,
        "final_checkpoint_id": parent_checkpoint_id,
        "resumed_from_checkpoint": resume_checkpoint_id or "",
        "world_branch": config.checkpoint_branch if registry else "",
        "elapsed_seconds": elapsed,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else 0.0
        ),
    }
