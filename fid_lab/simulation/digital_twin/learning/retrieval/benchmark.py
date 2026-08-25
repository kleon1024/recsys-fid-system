"""P3-06 equal-budget retrieval ladder on persisted v4 factual samples."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...observability import (
    FullFlowFixtureConfig,
    append_full_flow_partition,
    build_full_flow_fixtures,
    replace_json_atomic,
    verify_full_flow_dataset,
)
from ..contracts import Lane, learning_source_hash
from ..registry import PersistentModelRegistry
from ..sample_bus import PartitionedSampleBus
from .artifact import RetrievalANNIndex, RetrievalArtifact
from .contracts import DEFAULT_RETRIEVAL_FEATURE_CONTRACT, RetrievalModelConfig
from .data import corpus_from_snapshot, load_retrieval_batch
from .evaluation import (
    ann_exact_recall,
    graph_candidates,
    launch_decision,
    lifecycle_candidates,
    merge_candidate_sets,
    model_candidates,
    retrieval_metrics,
)
from .training import train_retrieval_model


@dataclass(frozen=True)
class RetrievalBenchmarkConfig:
    users: int = 10_000
    items: int = 100_000
    ticks: int = 4
    device: str = "cuda"
    top_k: int = 50
    downstream_k: int = 20
    route_k: int = 24
    merged_k: int = 96
    coarse_k: int = 48
    fine_k: int = 16
    expose_k: int = 8
    history_length: int = 32
    recall_negatives: int = 20
    epochs: int = 4
    batch_size: int = 1_024
    max_evaluation_queries: int = 10_000
    latency_budget_ms: float = 10.0
    seed: int = 2_026_082_5

    def __post_init__(self) -> None:
        numeric = asdict(self)
        if any(value <= 0 for name, value in numeric.items() if name != "device"):
            raise ValueError("retrieval benchmark configuration must be positive")
        if self.ticks < 2:
            raise ValueError("retrieval benchmark needs train and evaluation ticks")
        if not self.merged_k >= self.coarse_k >= self.fine_k >= self.expose_k:
            raise ValueError("retrieval benchmark cascade budgets differ")


def _evaluation_sample(batch, maximum: int, seed: int):
    if len(batch.request_id) <= maximum:
        return batch
    key = torch.remainder(batch.request_id * 1_000_003 + seed, 2_147_483_647)
    return batch.select(torch.argsort(key, stable=True)[:maximum])


def _frequency(batch, items: int) -> np.ndarray:
    return np.bincount(batch.positive_item_id.numpy(), minlength=items)


def _sample_diagnostics(train, evaluation, items: int, top_k: int) -> dict[str, object]:
    frequency = _frequency(train, items)
    target = evaluation.positive_item_id.numpy()
    popular = np.argsort(-frequency, kind="stable")[:top_k]
    return {
        "train_positive_unique_items": int((frequency > 0).sum()),
        "evaluation_positive_unique_items": int(np.unique(target).size),
        "evaluation_positive_seen_in_train_rate": float((frequency[target] > 0).mean()),
        "evaluation_positive_in_train_top_k_rate": float(np.isin(target, popular).mean()),
        "conditioning": "positive target is selected only from old-policy exposures",
        "randomized_retrieval_evidence": False,
        "interpretation": (
            "High old-policy Top-K coverage is a logging-policy ceiling. Novel learned "
            "candidates require randomized retrieval traffic or factual A/B evidence; "
            "the simulator and gate must not be tuned merely to make a neural model win."
        ),
    }


def _peak_cuda_gib(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0


def _prepare_inputs(config: RetrievalBenchmarkConfig, output: Path) -> dict[str, object]:
    started = perf_counter()
    snapshots = build_full_flow_fixtures(FullFlowFixtureConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        route_k=config.route_k,
        merged_k=config.merged_k,
        coarse_k=config.coarse_k,
        fine_k=config.fine_k,
        expose_k=config.expose_k,
        history_length=config.history_length,
        recall_negatives=config.recall_negatives,
        scenario="feed_consumption",
    ), ticks=config.ticks)
    generation_seconds = perf_counter() - started
    dataset_root = output / "dataset"
    materialize_started = perf_counter()
    partitions = tuple(
        append_full_flow_partition(snapshot, dataset_root, f"event_time={tick}")
        for tick, snapshot in enumerate(snapshots)
    )
    materialize_seconds = perf_counter() - materialize_started
    dataset = verify_full_flow_dataset(dataset_root)
    bus = PartitionedSampleBus(dataset_root, output / "lane-state")
    refs = bus.poll(Lane.CANDIDATE)
    train = load_retrieval_batch(bus, refs[:-1])
    evaluation = _evaluation_sample(
        load_retrieval_batch(bus, refs[-1:]),
        config.max_evaluation_queries,
        config.seed,
    )
    corpus = corpus_from_snapshot(snapshots[-1])
    corpus_file_sha256 = corpus.save(output / "corpus.pt")
    return {
        "snapshots": snapshots,
        "dataset": dataset,
        "partitions": partitions,
        "bus": bus,
        "refs": refs,
        "train": train,
        "evaluation": evaluation,
        "corpus": corpus,
        "corpus_file_sha256": corpus_file_sha256,
        "generation_seconds": generation_seconds,
        "materialize_seconds": materialize_seconds,
    }


def _baseline_metrics(train, evaluation, corpus, config) -> tuple[dict, np.ndarray, np.ndarray]:
    baseline_started = perf_counter()
    lifecycle = lifecycle_candidates(train, evaluation, corpus, config.top_k)
    lifecycle_seconds = perf_counter() - baseline_started
    graph_started = perf_counter()
    graph = graph_candidates(train, evaluation, lifecycle, config.top_k)
    graph_seconds = perf_counter() - graph_started
    merge_started = perf_counter()
    baseline = merge_candidate_sets((lifecycle, graph), config.top_k)
    merge_seconds = perf_counter() - merge_started
    frequency = _frequency(train, len(corpus.item_id))
    baseline_metrics = {
        "lifecycle_rules": retrieval_metrics(
            lifecycle, evaluation, corpus,
            downstream_k=config.downstream_k,
            train_frequency=frequency,
        ),
        "co_visit_graph": retrieval_metrics(
            graph, evaluation, corpus,
            downstream_k=config.downstream_k,
            baseline_candidates=lifecycle,
            train_frequency=frequency,
        ),
        "baseline_rrf": retrieval_metrics(
            baseline, evaluation, corpus,
            downstream_k=config.downstream_k,
            baseline_candidates=lifecycle,
            train_frequency=frequency,
        ),
    }
    lifecycle_latency = lifecycle_seconds * 1_000.0 / len(evaluation.request_id)
    graph_latency = graph_seconds * 1_000.0 / len(evaluation.request_id)
    rrf_latency = (
        lifecycle_seconds + graph_seconds + merge_seconds
    ) * 1_000.0 / len(evaluation.request_id)
    baseline_metrics["lifecycle_rules"].update({
        "milliseconds_per_query": lifecycle_latency,
        "decision": "control",
        "reason": "frozen lifecycle retrieval baseline",
    })
    for name, latency in (("co_visit_graph", graph_latency), ("baseline_rrf", rrf_latency)):
        decision, reason = launch_decision(
            baseline_metrics[name],
            milliseconds_per_query=latency,
            latency_budget_ms=config.latency_budget_ms,
            baseline_fixed_recall=float(
                baseline_metrics["lifecycle_rules"]["fixed_ranker_recall_at_k"]
            ),
        )
        baseline_metrics[name].update({
            "milliseconds_per_query": latency,
            "decision": decision,
            "reason": reason,
        })
    return baseline_metrics, baseline, frequency


def _evaluate_architecture(
    architecture: str,
    offset: int,
    *,
    train,
    evaluation,
    corpus,
    baseline,
    frequency,
    baseline_metrics,
    bus,
    registry,
    config,
    device,
) -> dict[str, object]:
    model_config = RetrievalModelConfig(
        architecture=architecture,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + offset,
    )
    model, training = train_retrieval_model(
        train, corpus, model_config, device=device,
    )
    artifact = RetrievalArtifact(
        model=model,
        config=model_config,
        feature_manifest_hash=train.feature_manifest_hash,
        retrieval_feature_contract_hash=DEFAULT_RETRIEVAL_FEATURE_CONTRACT.manifest_hash,
        corpus_sha256=corpus.content_sha256,
        training_report=training,
    )
    compatibility = bus.compatibility(
        index_version=artifact.index_version,
        corpus_sha256=corpus.content_sha256,
        stage_contract_hash=artifact.retrieval_feature_contract_hash,
    )
    record = registry.register_candidate(
        artifact, compatibility, lane=Lane.CANDIDATE,
        data_watermark=train.event_watermark,
    )
    loaded_record, loaded = registry.load(
        "candidate", compatibility, corpus=corpus,
    )
    if loaded_record.serving_version_id != record.serving_version_id:
        raise AssertionError("retrieval registry loaded another candidate")
    index = RetrievalANNIndex(loaded, corpus, device=device)
    candidates, latency = model_candidates(
        loaded, index, evaluation, top_k=config.top_k,
        batch_size=config.batch_size,
    )
    metrics = retrieval_metrics(
        candidates, evaluation, corpus, downstream_k=config.downstream_k,
        baseline_candidates=baseline, train_frequency=frequency,
    )
    decision, reason = launch_decision(
        metrics, milliseconds_per_query=latency,
        latency_budget_ms=config.latency_budget_ms,
        baseline_fixed_recall=float(
            baseline_metrics["baseline_rrf"]["fixed_ranker_recall_at_k"]
        ),
    )
    registry.shadow(record.serving_version_id, validation_status=decision)
    return {
        **metrics,
        "milliseconds_per_query": latency,
        "index_build_seconds": index.build_seconds,
        "ann_backend": index.backend,
        "ann_nlist": index.nlist,
        "ann_nprobe": index.nprobe,
        "ann_recall_at_k_vs_exact": ann_exact_recall(
            loaded, index, evaluation, top_k=config.top_k,
        ),
        "index_items": len(index.item_ids),
        "embedding_bytes": index.item_embeddings.nbytes,
        "index_version": artifact.index_version,
        "serving_version_id": record.serving_version_id,
        "artifact_sha256": record.artifact_sha256,
        "training": training,
        "decision": decision,
        "reason": reason,
    }


def run_retrieval_benchmark(
    config: RetrievalBenchmarkConfig,
    output: Path,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("retrieval benchmark output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    prepared = _prepare_inputs(config, output)
    train = prepared["train"]
    evaluation = prepared["evaluation"]
    corpus = prepared["corpus"]
    bus = prepared["bus"]
    refs = prepared["refs"]
    baseline_metrics, baseline, frequency = _baseline_metrics(
        train, evaluation, corpus, config,
    )
    registry = PersistentModelRegistry(output / "registry")
    models = {
        architecture: _evaluate_architecture(
            architecture, offset, train=train, evaluation=evaluation,
            corpus=corpus, baseline=baseline, frequency=frequency,
            baseline_metrics=baseline_metrics, bus=bus, registry=registry,
            config=config, device=device,
        )
        for offset, architecture in enumerate(("two_tower", "multi_interest"))
    }
    for ref in refs:
        bus.commit(Lane.CANDIDATE, ref)
    report = {
        "schema": "p3-fixed-budget-retrieval-ladder-v1",
        "quality_claim": False,
        "evidence_scope": "offline_and_serving_shadow_only_no_ab_lift",
        "config": asdict(config),
        "dataset_content_sha256": prepared["dataset"]["dataset_content_sha256"],
        "dataset_contract_hash": bus.contract_hash,
        "learning_source_sha256": learning_source_hash(),
        "feature_manifest_hash": train.feature_manifest_hash,
        "retrieval_feature_contract": asdict(DEFAULT_RETRIEVAL_FEATURE_CONTRACT),
        "retrieval_feature_contract_hash": (
            DEFAULT_RETRIEVAL_FEATURE_CONTRACT.manifest_hash
        ),
        "corpus_sha256": corpus.content_sha256,
        "corpus_file_sha256": prepared["corpus_file_sha256"],
        "partitions": list(prepared["partitions"]),
        "train_queries": len(train.request_id),
        "evaluation_queries": len(evaluation.request_id),
        "sample_diagnostics": _sample_diagnostics(
            train, evaluation, len(corpus.item_id), config.top_k,
        ),
        "same_corpus_top_k_latency_budget": True,
        "baseline": baseline_metrics,
        "models": models,
        "candidate_lane_cursor": bus.cursor(Lane.CANDIDATE).manifest(),
        "checkpoint_records": [
            asdict(record) for record in registry.checkpoint_records(config.ticks)
        ],
        "timing_seconds": {
            "generation": prepared["generation_seconds"],
            "materialization": prepared["materialize_seconds"],
            "total": perf_counter() - started,
        },
        "peak_cuda_gib": _peak_cuda_gib(device),
        "evidence_boundary": (
            "P3-06 equal-budget offline and serving-shadow retrieval evidence. "
            "P3-09 paired factual A/B remains required before promotion."
        ),
    }
    replace_json_atomic(output / "retrieval-leaderboard.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-evaluation-queries", type=int, default=10_000)
    args = parser.parse_args()
    report = run_retrieval_benchmark(RetrievalBenchmarkConfig(
        users=args.users,
        items=args.items,
        ticks=args.ticks,
        device=args.device,
        epochs=args.epochs,
        max_evaluation_queries=args.max_evaluation_queries,
    ), args.output)
    print((args.output / "retrieval-leaderboard.json").read_text())


if __name__ == "__main__":
    main()
