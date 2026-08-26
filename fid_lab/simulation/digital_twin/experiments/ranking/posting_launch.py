"""Creator-clustered Launch Review for the Posting recommendation funnel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time

import torch

from ...contracts import EventType, Surface
from ...engine import ExperimentPlan
from ...learning.probe import ProbeArtifact
from ...observability.store import replace_json_atomic
from ..launch_review import LaunchEvidenceCollector
from ..launch_review.metrics import _estimate
from ..retrieval_ladder import RetrievalLadderConfig, _build_kernel, _policy


@dataclass(frozen=True)
class PostingRankLaunchConfig:
    candidate_fine_checkpoint: str
    output: str
    users: int = 10_000
    items: int = 100_000
    burn_in_steps: int = 112
    experiment_steps: int = 128
    ticks_per_day: int = 16
    seed: int = 1_809
    device: str = "cuda"
    minimum_triggered_creators: int = 100


def _creator_metrics(events, request_assignment, triggered):
    metrics = {}
    event_types = {
        "click": EventType.CLICK,
        "create": EventType.CREATE,
        "publish": EventType.PUBLISH,
        "publish_failed": EventType.PUBLISH_FAILED,
    }
    values = {
        name: {0: {creator: 0.0 for creator in triggered[0]},
               1: {creator: 0.0 for creator in triggered[1]}}
        for name in event_types
    }
    published_posts: dict[int, tuple[int, int]] = {}
    for row in range(len(events.event_id)):
        assignment = request_assignment.get(int(events.request_id[row]))
        if assignment is None:
            continue
        creator, cell = assignment
        for name, event_type in event_types.items():
            if int(events.event_type[row]) == int(event_type):
                values[name][cell][creator] += 1.0
        if int(events.event_type[row]) == int(EventType.PUBLISH):
            post_id = int(events.post_id[row])
            if post_id >= 0:
                published_posts[post_id] = (creator, cell)
    qualified = {
        0: {creator: 0.0 for creator in triggered[0]},
        1: {creator: 0.0 for creator in triggered[1]},
    }
    for row in range(len(events.event_id)):
        if int(events.event_type[row]) != int(EventType.LONG_VIEW):
            continue
        assignment = published_posts.get(int(events.item_id[row]))
        if assignment is not None:
            creator, cell = assignment
            qualified[cell][creator] += 1.0
    values["early_qualified_long_view"] = qualified
    for name, cells in values.items():
        metrics[name] = _estimate(
            torch.tensor(tuple(cells[0].values()), dtype=torch.float32),
            torch.tensor(tuple(cells[1].values()), dtype=torch.float32),
        )
    return metrics


def _decision(metrics, sample, minimum):
    if min(sample.values()) < minimum:
        return "hold", "triggered-creator sample is below the gate"
    publish = metrics["publish"]
    failure = metrics["publish_failed"]
    if not all(
        math.isfinite(value)
        for metric in metrics.values()
        for value in metric.values()
    ):
        return "hold", "creator-clustered metrics contain non-finite values"
    if publish["ci95_high"] < 0.0:
        return "reject", "creator-level Publish significantly decreases"
    if failure["ci95_low"] > 0.0:
        return "reject", "Publish failures significantly increase"
    if publish["ci95_low"] <= 0.0:
        return "hold", "creator-level Publish confidence interval crosses zero"
    return "promote", "creator-level Publish improves without failure harm"


def run_posting_rank_launch(config: PostingRankLaunchConfig):
    started = time.perf_counter()
    runtime = RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        ticks_per_day=config.ticks_per_day,
        seed=config.seed,
        device=config.device,
    )
    _, kernel = _build_kernel(runtime)
    checkpoint = torch.load(
        config.candidate_fine_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    artifact = ProbeArtifact.from_checkpoint(checkpoint)
    kernel.platform.install_fine_scorer(1, artifact)
    control = _policy(
        "posting-formula-v1", 1, ("random", "popular"), config.ticks_per_day,
    )
    treatment = replace(
        control, name="posting-conditional-funnel-v1", fine_version_id=1,
    )
    baseline = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=control,
        experiment_seed=config.seed + 101,
        control_fraction=0.5,
        treatment_fraction=0.5,
        assignment_unit="creator",
        eligible_surfaces=(int(Surface.POSTING),),
    )
    logical_time = 0
    for _ in range(config.burn_in_steps):
        kernel.step(logical_time, baseline)
        logical_time += 1
    experiment = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=config.seed + 211,
        control_fraction=0.5,
        treatment_fraction=0.5,
        assignment_unit="creator",
        eligible_surfaces=(int(Surface.POSTING),),
    )
    start = logical_time
    evidence = LaunchEvidenceCollector()
    request_assignment: dict[int, tuple[int, int]] = {}
    triggered = {0: set(), 1: set()}
    for _ in range(config.experiment_steps):
        tick = kernel.step(logical_time, experiment)
        evidence.append(tick)
        trace = tick.candidate_trace
        if trace is not None:
            selected = (
                (trace.surface == int(Surface.POSTING))
                & torch.isin(
                    trace.experiment_cell,
                    torch.tensor((0, 1), device=trace.experiment_cell.device),
                )
                & (trace.user_creator_id >= 0)
            )
            for request, creator, cell in zip(
                trace.request_id[selected].tolist(),
                trace.user_creator_id[selected].tolist(),
                trace.experiment_cell[selected].tolist(),
                strict=True,
            ):
                request_assignment[int(request)] = (int(creator), int(cell))
                triggered[int(cell)].add(int(creator))
        logical_time += 1
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    events = events.select(events.ingest_time >= start)
    metrics = _creator_metrics(events, request_assignment, triggered)
    sample = {
        "control_triggered_creators": len(triggered[0]),
        "treatment_triggered_creators": len(triggered[1]),
    }
    decision, reason = _decision(
        metrics, sample, config.minimum_triggered_creators,
    )
    review = {
        "launch_review": "P-LR-001",
        "analysis_start_time": start,
        "analysis_end_time": logical_time - 1,
        "assignment_unit": "creator",
        "changed_owner": "Posting fine ranker only",
        "sample": sample,
        "metrics_per_triggered_creator": metrics,
        "decision": decision,
        "reason": reason,
    }
    review["launch_bundle"] = evidence.materialize(
        kernel=kernel,
        output_dir=Path(config.output) / "P-LR-001",
        review=review,
        ticks_per_day=config.ticks_per_day,
    )
    report = {
        "schema": "posting-ranker-launch/v1",
        "quality_claim": "synthetic factual-world evidence only",
        "config": asdict(config),
        "review": review,
        "elapsed_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(Path(config.output) / "report.json", report)
    return report
