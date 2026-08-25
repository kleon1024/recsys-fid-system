"""Train retrieval challengers on the evolving main-world request dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import torch

from ...checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ...experiments.retrieval_ladder import RetrievalLadderConfig, _build_kernel
from ...observability import replace_json_atomic, verify_full_flow_dataset
from ..contracts import Lane, learning_source_hash
from ..registry import PersistentModelRegistry
from ..sample_bus import PartitionedSampleBus
from .benchmark import (
    RetrievalBenchmarkConfig,
    _baseline_metrics,
    _evaluate_architecture,
    _evaluation_sample,
    _peak_cuda_gib,
    _sample_diagnostics,
)
from .contracts import DEFAULT_RETRIEVAL_FEATURE_CONTRACT
from .data import corpus_from_runtime, load_retrieval_batch


@dataclass(frozen=True)
class FactualRetrievalBenchmarkConfig:
    checkpoint_root: str
    dataset_root: str
    output: str
    checkpoint_branch: str = "main"
    users: int = 20_000
    items: int = 500_000
    device: str = "cuda"
    seed: int = 809
    ticks_per_day: int = 8
    evaluation_partitions: int = 4
    epochs: int = 4
    batch_size: int = 1_024
    top_k: int = 50
    downstream_k: int = 20
    max_evaluation_queries: int = 10_000
    latency_budget_ms: float = 10.0
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False

    def __post_init__(self) -> None:
        if min(
            self.users, self.items, self.ticks_per_day,
            self.evaluation_partitions, self.epochs, self.batch_size,
            self.top_k, self.downstream_k,
        ) <= 0:
            raise ValueError("factual retrieval benchmark dimensions must be positive")


def _benchmark_config(
    config: FactualRetrievalBenchmarkConfig,
    partitions: int,
) -> RetrievalBenchmarkConfig:
    return RetrievalBenchmarkConfig(
        users=config.users,
        items=config.items,
        ticks=partitions,
        device=config.device,
        top_k=config.top_k,
        downstream_k=config.downstream_k,
        epochs=config.epochs,
        batch_size=config.batch_size,
        max_evaluation_queries=config.max_evaluation_queries,
        latency_budget_ms=config.latency_budget_ms,
        daily_ticks=config.ticks_per_day,
        candidate_mode="additive",
        seed=config.seed,
    )


def run_factual_retrieval_benchmark(
    config: FactualRetrievalBenchmarkConfig,
) -> dict[str, object]:
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("factual retrieval benchmark output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _, kernel = _build_kernel(RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
    ))
    checkpoint_store = WorldCheckpointStore(Path(config.checkpoint_root))
    branch = WorldBranchRegistry(checkpoint_store).get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("diagnostic world data cannot train retrieval")
    checkpoint_store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=config.allow_additive_runtime_migration,
    )
    dataset = verify_full_flow_dataset(Path(config.dataset_root))
    bus = PartitionedSampleBus(Path(config.dataset_root), output / "lane-state")
    refs = bus.poll(Lane.CANDIDATE)
    if len(refs) <= config.evaluation_partitions:
        raise ValueError("factual retrieval split lacks training partitions")
    train_refs = refs[:-config.evaluation_partitions]
    evaluation_refs = refs[-config.evaluation_partitions:]
    train = load_retrieval_batch(bus, train_refs)
    evaluation = _evaluation_sample(
        load_retrieval_batch(bus, evaluation_refs),
        config.max_evaluation_queries,
        config.seed,
    )
    contract = bus.contract()["trace_manifest"]
    corpus = corpus_from_runtime(
        kernel.world.catalog,
        kernel.platform.projection.snapshot(),
        str(contract["catalog_version"]),
    )
    corpus_file_sha256 = corpus.save(output / "corpus.pt")
    benchmark = _benchmark_config(config, len(refs))
    baseline_metrics, baseline, frequency = _baseline_metrics(
        train, evaluation, corpus, benchmark,
    )
    registry = PersistentModelRegistry(output / "registry")
    started = perf_counter()
    models = {
        architecture: _evaluate_architecture(
            architecture,
            offset,
            train=train,
            evaluation=evaluation,
            corpus=corpus,
            baseline=baseline,
            frequency=frequency,
            baseline_metrics=baseline_metrics,
            bus=bus,
            registry=registry,
            config=benchmark,
            device=device,
        )
        for offset, architecture in enumerate(("two_tower", "multi_interest"))
    }
    diagnostics = _sample_diagnostics(
        train, evaluation, len(corpus.item_id), config.top_k,
    )
    if (
        not diagnostics["randomized_retrieval_evidence"]
        and diagnostics["evaluation_positive_in_train_top_k_rate"] > 0.95
    ):
        for model in models.values():
            model["observational_decision"] = model["decision"]
            model["observational_reason"] = model["reason"]
            model["decision"] = "hold_randomized_support_required"
            model["reason"] = (
                "old-policy exposure labels cannot identify incremental "
                "candidate relevance; randomized retrieval traffic is required"
            )
    for ref in refs:
        bus.commit(Lane.CANDIDATE, ref)
    feature_contract = benchmark.daily_ticks
    report = {
        "schema": "factual-p3-retrieval-shadow/v1",
        "quality_claim": "offline and serving shadow only; factual A/B required",
        "config": asdict(config),
        "world_checkpoint_id": branch.head_checkpoint_id,
        "dataset_content_sha256": dataset["dataset_content_sha256"],
        "dataset_contract_hash": bus.contract_hash,
        "learning_source_sha256": learning_source_hash(),
        "feature_manifest_hash": train.feature_manifest_hash,
        "retrieval_feature_contract": {
            **asdict(DEFAULT_RETRIEVAL_FEATURE_CONTRACT),
            "daily_ticks": feature_contract,
        },
        "corpus_sha256": corpus.content_sha256,
        "corpus_file_sha256": corpus_file_sha256,
        "train_partitions": [ref.key for ref in train_refs],
        "evaluation_partitions": [ref.key for ref in evaluation_refs],
        "train_queries": len(train.request_id),
        "evaluation_queries": len(evaluation.request_id),
        "sample_diagnostics": diagnostics,
        "same_corpus_top_k_latency_budget": True,
        "baseline": baseline_metrics,
        "models": models,
        "candidate_lane_cursor": bus.cursor(Lane.CANDIDATE).manifest(),
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_gib": _peak_cuda_gib(device),
    }
    replace_json_atomic(output / "retrieval-leaderboard.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-branch", default="main")
    parser.add_argument("--users", type=int, default=20_000)
    parser.add_argument("--items", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--ticks-per-day", type=int, default=8)
    parser.add_argument("--evaluation-partitions", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument("--allow-additive-runtime-migration", action="store_true")
    args = parser.parse_args()
    report = run_factual_retrieval_benchmark(FactualRetrievalBenchmarkConfig(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        output=args.output,
        checkpoint_branch=args.checkpoint_branch,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
        evaluation_partitions=args.evaluation_partitions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        allow_code_migration=args.allow_code_migration,
        allow_additive_runtime_migration=args.allow_additive_runtime_migration,
    ))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
