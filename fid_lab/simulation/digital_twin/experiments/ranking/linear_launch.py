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
from ...learning.probe import ProbeArtifact, load_probe_batch, train_probe
from ...observability.store import replace_json_atomic
from ...value_tree import FEED_VALUE_TREE_VERSION, task_value_weights
from ..launch_review import LaunchEvidenceCollector
from ..launch_review.metrics import analyze_experiment, decide_launch, validate_aa
from ..retrieval_ladder import RetrievalLadderConfig, _build_kernel, _policy
from .window import run_window


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
    control_fine_checkpoint: str = ""
    replay_dataset_root: str = ""
    replay_partition_fraction: float = 0.20
    run_aa: bool = False
    candidate_fine_checkpoint: str = ""
    candidate_stay_weight: float | None = None


_ROW_TENSORS = (
    "request_id", "user_id", "surface", "request_time", "item_id",
    "position", "route_id", "recall_score", "exposed",
    "candidate_exposure_probability", "randomized_support", "dwell_ms",
    "dense_features", "sparse_buckets", "labels", "label_mask",
    "label_applicable", "label_mature", "joint_logging_probability",
)


def _append_replay(current, replay):
    if (
        current.task_names != replay.task_names
        or current.feature_manifest_hash != replay.feature_manifest_hash
    ):
        raise ValueError("replay sample contract differs from current samples")
    return replace(
        current,
        **{
            name: torch.cat((getattr(current, name), getattr(replay, name)))
            for name in _ROW_TENSORS
        },
        partition_content_hashes=(
            current.partition_content_hashes + replay.partition_content_hashes
        ),
        event_watermark=max(current.event_watermark, replay.event_watermark),
    )


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
    if not 0.0 < config.replay_partition_fraction <= 1.0:
        raise ValueError("replay partition fraction must be in (0, 1]")
    state = Path(config.output) / "training-lane"
    bus = PartitionedSampleBus(Path(config.dataset_root), state)
    refs = bus.poll(Lane.CANDIDATE)
    if len(refs) < 5:
        raise ValueError("linear rank launch requires at least five partitions")
    split = max(1, int(0.8 * len(refs)))
    train = load_probe_batch(bus, refs[:split])
    validation = load_probe_batch(bus, refs[split:])
    current_train_rows = len(train.request_id)
    replay_rows = 0
    if config.replay_dataset_root:
        replay_bus = PartitionedSampleBus(
            Path(config.replay_dataset_root), state / "replay",
        )
        replay_refs = replay_bus.poll(Lane.CANDIDATE)
        replay_count = max(
            1, int(len(replay_refs) * config.replay_partition_fraction),
        )
        replay = load_probe_batch(replay_bus, replay_refs[-replay_count:])
        replay_rows = len(replay.request_id)
        train = _append_replay(train, replay)
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
    initial_artifact = None
    if config.control_fine_checkpoint:
        checkpoint = torch.load(
            config.control_fine_checkpoint, map_location="cpu", weights_only=True,
        )
        initial_artifact = ProbeArtifact.from_checkpoint(checkpoint)
    artifact = train_probe(
        train,
        lane=Lane.CANDIDATE,
        initial_artifact=initial_artifact,
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
        probability = artifact.predict_task_probabilities(
            dense,
            validation.surface[mask].to(parameter.device),
        )[:, task].cpu()
    labels = validation.labels[mask, task]
    loss = torch.nn.functional.binary_cross_entropy(probability, labels)
    labeled = train.label_mask.any(dim=1)
    return artifact, {
        "train_rows": len(train.request_id),
        "validation_rows": len(validation.request_id),
        "validation_long_view_rows": int(mask.sum()),
        "validation_long_view_auc": _auc(labels, probability),
        "validation_long_view_gauc": _gauc(
            validation.request_id[mask], labels, probability,
        ),
        "validation_long_view_logloss": float(loss),
        "training_support": {
            "current_train_rows": current_train_rows,
            "historical_replay_rows": replay_rows,
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


def _evaluate_resumed_candidate(config: LinearRankLaunchConfig, artifact):
    bus = PartitionedSampleBus(
        Path(config.dataset_root), Path(config.output) / "resume-evaluation-lane",
    )
    refs = bus.poll(Lane.CANDIDATE)
    split = max(1, int(0.8 * len(refs)))
    validation = load_probe_batch(bus, refs[split:])
    task = validation.task_names.index("long_view")
    mask = validation.label_mask[:, task] & (
        validation.surface == int(Surface.FEED)
    )
    with torch.inference_mode():
        probability = artifact.predict_task_probabilities(
            validation.dense_features[mask], validation.surface[mask],
        )[:, task].cpu()
    labels = validation.labels[mask, task]
    return {
        "train_rows": int(artifact.training_report["rows"]),
        "validation_rows": len(validation.request_id),
        "validation_long_view_rows": int(mask.sum()),
        "validation_long_view_auc": _auc(labels, probability),
        "validation_long_view_gauc": _gauc(
            validation.request_id[mask], labels, probability,
        ),
        "validation_long_view_logloss": float(
            torch.nn.functional.binary_cross_entropy(probability, labels)
        ),
        "training_support": {
            "resumed_from_candidate_checkpoint": config.candidate_fine_checkpoint,
        },
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


def run_linear_rank_launch(config: LinearRankLaunchConfig) -> dict[str, object]:
    started = time.perf_counter()
    if config.candidate_fine_checkpoint:
        checkpoint = torch.load(
            config.candidate_fine_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        artifact = ProbeArtifact.from_checkpoint(checkpoint)
        if config.candidate_stay_weight is not None:
            if config.candidate_stay_weight < 0.0:
                raise ValueError("candidate stay weight must be non-negative")
            weights = list(artifact.serving_task_weights or ())
            if not weights or "stay_value" not in artifact.task_names:
                raise ValueError("candidate artifact has no Stay value target")
            weights[artifact.task_names.index("stay_value")] = (
                config.candidate_stay_weight
            )
            artifact = replace(artifact, serving_task_weights=tuple(weights))
        offline = _evaluate_resumed_candidate(config, artifact)
    else:
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
    control_version = 0
    control_name = "feed-random-popular-formula"
    if config.control_fine_checkpoint:
        checkpoint = torch.load(
            config.control_fine_checkpoint, map_location="cpu", weights_only=True,
        )
        kernel.platform.install_fine_scorer(
            1, ProbeArtifact.from_checkpoint(checkpoint),
        )
        control_version = 1
        control_name = "feed-random-popular-accepted-vt"
    control = _policy(
        control_name, 1, ("random", "popular"),
        config.ticks_per_day,
    )
    control = replace(control, fine_version_id=control_version)
    baseline = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=control,
        experiment_seed=config.seed + 101,
        control_fraction=0.5,
        treatment_fraction=0.5,
        eligible_surfaces=(int(Surface.FEED),),
    )
    logical_time, _ = run_window(
        kernel, baseline, 0, config.burn_in_steps,
    )
    aa_metrics: dict[str, object] = {}
    aa_sample: dict[str, object] = {}
    aa_valid = True
    aa_reason = "handled by the experiment-platform health monitor"
    if config.run_aa:
        logical_time, aa_events = run_window(
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
    candidate_version = control_version + 1
    kernel.platform.install_fine_scorer(candidate_version, artifact)
    treatment = replace(
        control,
        name="feed-random-popular-dense-lr-stay-v2",
        fine_version_id=candidate_version,
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
    logical_time, events = run_window(
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
        "control_fine_version": control_version,
        "treatment_fine_version": candidate_version,
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
            "run_for_this_launch": config.run_aa,
        },
        "review": review,
        "elapsed_seconds": time.perf_counter() - started,
    }
    replace_json_atomic(output / "report.json", report)
    return report
