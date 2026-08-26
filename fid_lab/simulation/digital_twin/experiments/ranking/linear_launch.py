"""Train the first factual dense LR and run formula-versus-model A/B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time

import torch

from ...contracts import Surface
from ...engine import ExperimentPlan
from ...learning import Lane, PartitionedSampleBus
from ...learning.probe import load_probe_batch, train_probe
from ...observability.store import replace_json_atomic
from ..launch_review import LaunchEvidenceCollector
from ..launch_review.metrics import analyze_experiment, decide_launch
from ..retrieval_ladder import RetrievalLadderConfig, _build_kernel, _policy


@dataclass(frozen=True)
class LinearRankLaunchConfig:
    dataset_root: str
    output: str
    users: int = 10_000
    items: int = 100_000
    burn_in_steps: int = 112
    aa_steps: int = 32
    experiment_steps: int = 64
    ticks_per_day: int = 16
    seed: int = 1_809
    device: str = "cuda"
    epochs: int = 4
    minimum_triggered_users: int = 2_000


def _auc(label: torch.Tensor, score: torch.Tensor) -> float:
    positive = label > 0.5
    positives = int(positive.sum())
    negatives = len(label) - positives
    if not positives or not negatives:
        return float("nan")
    order = torch.argsort(score)
    rank = torch.empty_like(order, dtype=torch.float)
    rank[order] = torch.arange(1, len(label) + 1, dtype=torch.float)
    statistic = rank[positive].sum() - positives * (positives + 1) / 2
    return float(statistic / (positives * negatives))


def _train_candidate(config: LinearRankLaunchConfig):
    state = Path(config.output) / "training-lane"
    bus = PartitionedSampleBus(Path(config.dataset_root), state)
    refs = bus.poll(Lane.CANDIDATE)
    if len(refs) < 5:
        raise ValueError("linear rank launch requires at least five partitions")
    split = max(1, int(0.8 * len(refs)))
    train = load_probe_batch(bus, refs[:split])
    validation = load_probe_batch(bus, refs[split:])
    artifact = train_probe(
        train,
        lane=Lane.CANDIDATE,
        epochs=config.epochs,
        device=config.device,
        seed=config.seed + 71,
    )
    task = validation.task_names.index("long_view")
    mask = validation.label_mask[:, task] & (
        validation.surface == int(Surface.FEED)
    )
    with torch.inference_mode():
        parameter = next(artifact.model.parameters())
        logits = artifact.model(
            validation.dense_features[mask].to(parameter.device),
            validation.surface[mask].to(parameter.device),
        )[:, task].cpu()
    labels = validation.labels[mask, task]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return artifact, {
        "train_rows": len(train.request_id),
        "validation_rows": len(validation.request_id),
        "validation_long_view_rows": int(mask.sum()),
        "validation_long_view_auc": _auc(labels, logits),
        "validation_long_view_logloss": float(loss),
        "train_time_range": [
            int(train.request_time.min()), int(train.request_time.max()),
        ],
        "validation_time_range": [
            int(validation.request_time.min()), int(validation.request_time.max()),
        ],
        "artifact": artifact.training_report,
    }


def _run_window(kernel, plan, logical_time, steps, evidence=None):
    start = logical_time
    for _ in range(steps):
        tick = kernel.step(logical_time, plan)
        if evidence is not None:
            evidence.append(tick)
        logical_time += 1
    events = kernel.event_log.read(ingested_through=logical_time - 1)
    return logical_time, events.select(events.ingest_time >= start)


def run_linear_rank_launch(config: LinearRankLaunchConfig) -> dict[str, object]:
    started = time.perf_counter()
    artifact, offline = _train_candidate(config)
    runtime = RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        ticks_per_day=config.ticks_per_day,
        seed=config.seed,
        device=config.device,
        response_authority_mode="formula_oracle",
    )
    _, kernel = _build_kernel(runtime)
    control = _policy(
        "feed-random-popular-formula", 1, ("random", "popular"),
        config.ticks_per_day,
    )
    baseline = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=control,
        experiment_seed=config.seed + 101,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    logical_time, _ = _run_window(
        kernel, baseline, 0, config.burn_in_steps,
    )
    logical_time, aa_events = _run_window(
        kernel, baseline, logical_time, config.aa_steps,
    )
    aa_metrics, aa_sample = analyze_experiment(aa_events, config.users)
    kernel.platform.install_fine_scorer(1, artifact)
    treatment = replace(
        control, name="feed-random-popular-dense-lr-v1", fine_version_id=1,
    )
    experiment = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=config.seed + 211,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    evidence = LaunchEvidenceCollector()
    start = logical_time
    logical_time, events = _run_window(
        kernel, experiment, logical_time, config.experiment_steps, evidence,
    )
    metrics, sample = analyze_experiment(events, config.users)
    decision, reason = decide_launch(
        metrics, sample, config.minimum_triggered_users,
    )
    review = {
        "launch_review": "F-LR-001",
        "analysis_start_time": start,
        "analysis_end_time": logical_time - 1,
        "changed_owner": "fine ranker only",
        "control_fine_version": 0,
        "treatment_fine_version": 1,
        "sample": sample,
        "metrics_per_triggered_user": metrics,
        "decision": decision,
        "reason": reason,
    }
    output = Path(config.output)
    review["launch_bundle"] = evidence.materialize(
        kernel=kernel,
        output_dir=output / "F-LR-001",
        review=review,
        ticks_per_day=config.ticks_per_day,
    )
    report = {
        "schema": "dense-linear-ranker-launch/v1",
        "quality_claim": "synthetic factual-world evidence only",
        "config": asdict(config),
        "offline": offline,
        "aa": {
            "sample": aa_sample,
            "metrics_per_triggered_user": aa_metrics,
        },
        "review": review,
        "elapsed_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(output / "report.json", report)
    return report
