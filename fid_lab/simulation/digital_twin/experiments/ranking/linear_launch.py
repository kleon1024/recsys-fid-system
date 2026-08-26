"""Train the first factual dense LR and run formula-versus-model A/B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import torch

from ...contracts import Surface
from ...engine import ExperimentPlan
from ...learning import Lane, PartitionedSampleBus
from ...learning.probe import load_probe_batch, train_probe
from ...observability.store import replace_json_atomic
from ...value_tree import FEED_VALUE_TREE_VERSION, task_value_weights
from ..launch_review import LaunchEvidenceCollector
from ..launch_review.metrics import analyze_experiment, decide_launch, validate_aa
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
    epochs: int = 24
    learning_rate: float = 1e-2
    minimum_triggered_users: int = 2_000


def _auc(label: torch.Tensor, score: torch.Tensor) -> float:
    positive = label > 0.5
    positives = int(positive.sum())
    negatives = len(label) - positives
    if not positives or not negatives:
        return float("nan")
    order = torch.argsort(score)
    sorted_score = score[order]
    _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
    end = torch.cumsum(counts, dim=0).float()
    start = end - counts.float() + 1.0
    average_rank = torch.repeat_interleave((start + end) / 2.0, counts)
    rank = torch.empty_like(average_rank)
    rank[order] = average_rank
    statistic = rank[positive].sum() - positives * (positives + 1) / 2
    return float(statistic / (positives * negatives))


def _gauc(
    request_id: torch.Tensor,
    label: torch.Tensor,
    score: torch.Tensor,
) -> float:
    order = torch.argsort(request_id, stable=True)
    ordered_request = request_id[order]
    ordered_label = label[order]
    ordered_score = score[order]
    starts = torch.ones_like(ordered_request, dtype=torch.bool)
    starts[1:] = ordered_request[1:] != ordered_request[:-1]
    begin = torch.where(starts)[0]
    end = torch.cat((
        begin[1:], torch.tensor([len(order)], device=begin.device),
    ))
    weighted_auc = 0.0
    comparable_pairs = 0
    for left, right in zip(begin.tolist(), end.tolist(), strict=True):
        group_label = ordered_label[left:right]
        positives = int((group_label > 0.5).sum())
        negatives = len(group_label) - positives
        if not positives or not negatives:
            continue
        pairs = positives * negatives
        weighted_auc += pairs * _auc(
            group_label, ordered_score[left:right],
        )
        comparable_pairs += pairs
    return weighted_auc / comparable_pairs if comparable_pairs else float("nan")


def _train_candidate(config: LinearRankLaunchConfig):
    state = Path(config.output) / "training-lane"
    bus = PartitionedSampleBus(Path(config.dataset_root), state)
    refs = bus.poll(Lane.CANDIDATE)
    if len(refs) < 5:
        raise ValueError("linear rank launch requires at least five partitions")
    split = max(1, int(0.8 * len(refs)))
    train = load_probe_batch(bus, refs[:split])
    validation = load_probe_batch(bus, refs[split:])
    feed = train.surface == int(Surface.FEED)
    train = replace(
        train,
        request_id=train.request_id[feed],
        user_id=train.user_id[feed],
        surface=train.surface[feed],
        request_time=train.request_time[feed],
        item_id=train.item_id[feed],
        position=train.position[feed],
        route_id=train.route_id[feed],
        recall_score=train.recall_score[feed],
        exposed=train.exposed[feed],
        candidate_exposure_probability=(
            train.candidate_exposure_probability[feed]
        ),
        randomized_support=train.randomized_support[feed],
        dwell_ms=train.dwell_ms[feed],
        dense_features=train.dense_features[feed],
        sparse_buckets=train.sparse_buckets[feed],
        labels=train.labels[feed],
        label_mask=train.label_mask[feed],
        label_applicable=train.label_applicable[feed],
        label_mature=train.label_mature[feed],
        joint_logging_probability=train.joint_logging_probability[feed],
    )
    artifact = train_probe(
        train,
        lane=Lane.CANDIDATE,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=config.device,
        seed=config.seed + 71,
    )
    artifact = replace(
        artifact,
        serving_task_weights=task_value_weights(artifact.task_names),
    )
    task = validation.task_names.index("long_view")
    mask = validation.label_mask[:, task] & (
        validation.surface == int(Surface.FEED)
    )
    with torch.inference_mode():
        parameter = next(artifact.model.parameters())
        dense = validation.dense_features[mask].to(parameter.device)
        mean = artifact.dense_mean.to(parameter.device)
        scale = artifact.dense_scale.to(parameter.device)
        logits = artifact.model(
            (dense - mean) / scale,
            validation.surface[mask].to(parameter.device),
        )[:, task].cpu()
    labels = validation.labels[mask, task]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    labeled = train.label_mask.any(dim=1)
    return artifact, {
        "train_rows": len(train.request_id),
        "validation_rows": len(validation.request_id),
        "validation_long_view_rows": int(mask.sum()),
        "validation_long_view_auc": _auc(labels, logits),
        "validation_long_view_gauc": _gauc(
            validation.request_id[mask], labels, logits,
        ),
        "validation_long_view_logloss": float(loss),
        "training_support": {
            "labeled_rows": int(labeled.sum()),
            "randomized_rows": int(train.randomized_support.sum()),
            "randomized_labeled_rows": int(
                (labeled & train.randomized_support).sum()
            ),
            "exposed_rows": int(train.exposed.sum()),
            "candidate_rows": len(train.request_id),
        },
        "train_time_range": [
            int(train.request_time.min()), int(train.request_time.max()),
        ],
        "validation_time_range": [
            int(validation.request_time.min()), int(validation.request_time.max()),
        ],
        "artifact": artifact.training_report,
        "value_tree_version": FEED_VALUE_TREE_VERSION,
        "serving_task_weights": dict(zip(
            artifact.task_names,
            artifact.serving_task_weights or (),
            strict=True,
        )),
    }


def _save_serving_artifact(artifact, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "fine-ranker.pt"
    with NamedTemporaryFile(
        dir=output, prefix=".fine-ranker-", suffix=".pt", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(artifact.checkpoint(), temporary)
        digest = sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": digest,
        "model_name": artifact.model_name,
        "feature_manifest_hash": artifact.feature_manifest_hash,
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
    output = Path(config.output)
    serving_artifact = _save_serving_artifact(artifact, output)
    if offline["validation_long_view_auc"] < 0.52:
        report = {
            "schema": "dense-linear-ranker-launch/v1",
            "quality_claim": "synthetic factual-world evidence only",
            "config": asdict(config),
            "offline": offline,
            "serving_artifact": serving_artifact,
            "review": {
                "decision": "reject_offline",
                "reason": "time-split long-view AUC is below 0.52",
                "sample": {},
                "metrics_per_triggered_user": {},
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        replace_json_atomic(Path(config.output) / "report.json", report)
        return report
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
    aa_valid, aa_reason = validate_aa(aa_metrics)
    if not aa_valid:
        report = {
            "schema": "dense-linear-ranker-launch/v1",
            "quality_claim": "synthetic factual-world evidence only",
            "config": asdict(config),
            "offline": offline,
            "serving_artifact": serving_artifact,
            "aa": {
                "sample": aa_sample,
                "metrics_per_triggered_user": aa_metrics,
                "valid": False,
                "reason": aa_reason,
            },
            "review": {
                "decision": "invalid_aa",
                "reason": aa_reason,
                "sample": {},
                "metrics_per_triggered_user": {},
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        replace_json_atomic(output / "report.json", report)
        return report
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
        "serving_artifact": serving_artifact,
        "aa": {
            "sample": aa_sample,
            "metrics_per_triggered_user": aa_metrics,
            "valid": True,
            "reason": aa_reason,
        },
        "review": review,
        "elapsed_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(output / "report.json", report)
    return report
