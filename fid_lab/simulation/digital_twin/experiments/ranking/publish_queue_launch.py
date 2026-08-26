"""Train dense/sparse Publish Queue LR candidates and run a Feed A/B."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch

from ...contracts import Surface
from ...engine import ExperimentPlan
from ...learning import (
    Lane,
    PartitionedSampleBus,
    ProbeArtifact,
    SparseLinearArtifact,
    train_probe,
)
from ...learning.publish_queue import (
    PUBLISH_QUEUE_VALUE_VERSION,
    load_publish_queue_batch,
    publish_queue_task_weights,
)
from ...learning.sparse_linear import train_sparse_linear
from ...observability.store import replace_json_atomic
from ..launch_review import LaunchEvidenceCollector
from ..launch_review.metrics import analyze_experiment
from ..retrieval_ladder import RetrievalLadderConfig, _build_kernel, _policy
from .window import run_window


@dataclass(frozen=True)
class PublishQueueLaunchConfig:
    dataset_root: str
    control_fine_checkpoint: str
    output: str
    users: int = 10_000
    items: int = 100_000
    burn_in_steps: int = 112
    experiment_steps: int = 96
    ticks_per_day: int = 16
    seed: int = 1_809
    device: str = "cuda"
    epochs: int = 8
    learning_rate: float = 1e-2
    sparse_hash_size: int = 1 << 18
    publish_weight: float = 0.12
    minimum_triggered_users: int = 2_000


def _mature_partition_split(bus: PartitionedSampleBus):
    refs = bus.poll(Lane.CANDIDATE)
    windows = bus.contract()["sample_contract"][
        "publish_queue_task_window_ticks"
    ]
    maximum_window = max(int(value) for value in windows)
    mature = tuple(
        ref for ref in refs
        if ref.event_time_max <= ref.event_watermark - maximum_window
    )
    if len(mature) < 5:
        raise ValueError("Publish Queue requires five fully mature partitions")
    split = min(len(mature) - 1, max(1, int(0.8 * len(mature))))
    return mature[:split], mature[split:]


def _task_metrics(artifact, batch, task_name: str) -> dict[str, float | int]:
    task = batch.task_names.index(task_name)
    mask = batch.label_mask[:, task]
    if not mask.any():
        raise ValueError(f"validation has no mature {task_name} labels")
    parameter = next(artifact.model.parameters())
    with torch.inference_mode():
        if isinstance(artifact, SparseLinearArtifact):
            probability = artifact.predict_task_probabilities(
                batch.dense_features[mask].to(parameter.device),
                batch.sparse_buckets[mask].to(parameter.device),
                batch.surface[mask].to(parameter.device),
            )[:, task]
        else:
            probability = artifact.predict_task_probabilities(
                batch.dense_features[mask].to(parameter.device),
                batch.surface[mask].to(parameter.device),
            )[:, task]
    probability = probability.detach().cpu().numpy().clip(1e-6, 1 - 1e-6)
    label = (batch.labels[mask, task] > 0.0).numpy().astype(np.int64)
    if len(np.unique(label)) != 2:
        raise ValueError(f"validation {task_name} lacks both classes")
    return {
        "rows": int(mask.sum()),
        "positives": int(label.sum()),
        "auc": float(roc_auc_score(label, probability)),
        "pr_auc": float(average_precision_score(label, probability)),
        "logloss": float(log_loss(label, probability)),
    }


def _train_candidates(config: PublishQueueLaunchConfig):
    bus = PartitionedSampleBus(
        Path(config.dataset_root), Path(config.output) / "training-lane",
    )
    train_refs, validation_refs = _mature_partition_split(bus)
    train = load_publish_queue_batch(bus, train_refs)
    validation = load_publish_queue_batch(bus, validation_refs)
    dense = train_probe(
        train,
        lane=Lane.CANDIDATE,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=config.device,
        seed=config.seed + 301,
    )
    dense = replace(
        dense, serving_task_weights=publish_queue_task_weights(dense.task_names),
    )
    sparse = train_sparse_linear(
        train,
        lane=Lane.CANDIDATE,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        hash_size=config.sparse_hash_size,
        device=config.device,
        seed=config.seed + 307,
    )
    sparse = replace(
        sparse,
        serving_task_weights=publish_queue_task_weights(sparse.task_names),
    )
    reports = {
        "dense_lr": _task_metrics(dense, validation, "publish_48h"),
        "sparse_fid_lr": _task_metrics(sparse, validation, "publish_48h"),
    }
    selected_name = max(reports, key=lambda name: reports[name]["pr_auc"])
    selected = dense if selected_name == "dense_lr" else sparse
    return selected_name, selected, {
        "train_partitions": [ref.key for ref in train_refs],
        "validation_partitions": [ref.key for ref in validation_refs],
        "train_rows": len(train.request_id),
        "validation_rows": len(validation.request_id),
        "models": reports,
        "selected": selected_name,
        "value_version": PUBLISH_QUEUE_VALUE_VERSION,
        "task_weights": dict(zip(
            selected.task_names,
            selected.serving_task_weights or (),
            strict=True,
        )),
    }


def _save_artifact(artifact, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "publish-queue-ranker.pt"
    with NamedTemporaryFile(
        dir=output, prefix=".publish-queue-", suffix=".pt", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(artifact.checkpoint(), temporary)
        digest = sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": digest, "model": artifact.model_name}


def _publish_decision(metrics, sample, minimum_users):
    if min(sample.values()) < minimum_users:
        return "hold", "triggered-user sample is below the gate"
    if metrics["dwell_seconds"]["ci95_high"] < 0.0:
        return "reject", "stay significantly decreases"
    if metrics["negative"]["ci95_low"] > 0.0:
        return "reject", "negative feedback significantly increases"
    if metrics["publish"]["ci95_high"] <= 0.0:
        return "reject", "publication does not improve"
    if metrics["publish"]["ci95_low"] <= 0.0:
        return "hold", "publication confidence interval crosses zero"
    return "promote", "publication improves and Feed guardrails pass"


def run_publish_queue_launch(
    config: PublishQueueLaunchConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    selected_name, artifact, offline = _train_candidates(config)
    serving = _save_artifact(artifact, Path(config.output))
    runtime = RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        ticks_per_day=config.ticks_per_day,
        seed=config.seed,
        device=config.device,
        response_authority_mode="formula_oracle",
    )
    _, kernel = _build_kernel(runtime)
    fine = ProbeArtifact.from_checkpoint(torch.load(
        config.control_fine_checkpoint, map_location="cpu", weights_only=True,
    ))
    kernel.platform.install_fine_scorer(1, fine)
    kernel.platform.install_publish_scorer(1, artifact)
    control = replace(
        _policy(
            "feed-random-popular-accepted-vt", 1, ("random", "popular"),
            config.ticks_per_day,
        ),
        fine_version_id=1,
    )
    treatment = replace(
        control,
        name=f"feed-publish-{selected_name}",
        mix_version_id=control.mix_version_id + 1,
        publish_version_id=1,
        publish_weight=config.publish_weight,
    )
    baseline = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=control,
        experiment_seed=config.seed + 401,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    logical_time, _ = run_window(kernel, baseline, 0, config.burn_in_steps)
    experiment = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=config.seed + 409,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    evidence = LaunchEvidenceCollector()
    analysis_start = logical_time
    logical_time, events = run_window(
        kernel, experiment, logical_time, config.experiment_steps, evidence,
    )
    metrics, sample = analyze_experiment(events, config.users)
    decision, reason = _publish_decision(
        metrics, sample, config.minimum_triggered_users,
    )
    review = {
        "launch_review": "PUBLISH-LR-001",
        "analysis_start_time": analysis_start,
        "analysis_end_time": logical_time - 1,
        "changed_owner": "Feed Publish Queue score and mixer weight only",
        "sample": sample,
        "metrics_per_triggered_user": metrics,
        "decision": decision,
        "reason": reason,
    }
    review["launch_bundle"] = evidence.materialize(
        kernel=kernel,
        output_dir=Path(config.output) / "PUBLISH-LR-001",
        review=review,
        ticks_per_day=config.ticks_per_day,
    )
    report = {
        "schema": "feed-publish-queue-launch/v1",
        "quality_claim": "synthetic factual-world evidence only",
        "config": asdict(config),
        "offline": offline,
        "serving_artifact": serving,
        "review": review,
        "elapsed_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(Path(config.output) / "report.json", report)
    return report
