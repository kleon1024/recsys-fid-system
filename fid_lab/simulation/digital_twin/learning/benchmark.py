"""RTX-scale acceptance runner for the P3-04/05 infrastructure slice."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from ..observability import (
    FullFlowFixtureConfig,
    append_full_flow_partition,
    build_full_flow_fixtures,
    replace_json_atomic,
    verify_full_flow_dataset,
)
from .contracts import Lane
from .probe import feature_drift_report, load_probe_batch, train_probe
from .registry import PersistentModelRegistry
from .sample_bus import PartitionedSampleBus


@dataclass(frozen=True)
class LearningBenchmarkConfig:
    users: int = 10_000
    items: int = 100_000
    ticks: int = 2
    device: str = "cuda"
    merged_k: int = 96
    coarse_k: int = 48
    fine_k: int = 16
    expose_k: int = 8
    history_length: int = 32
    recall_negatives: int = 20

    def __post_init__(self) -> None:
        if min(self.users, self.items, self.ticks) <= 0:
            raise ValueError("benchmark scale must be positive")
        if not self.merged_k >= self.coarse_k >= self.fine_k >= self.expose_k:
            raise ValueError("benchmark cascade budgets are inconsistent")


def _catalog_hash(catalog) -> str:
    digest = sha256()
    for name in (
        "item_id", "content_kind", "topic_id", "content_embedding",
        "creator_id", "country", "region", "publish_time", "active",
    ):
        value = getattr(catalog, name).detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _peak_cuda_gib(device: torch.device) -> float:
    return (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda" else 0.0
    )


def _max_drift(report: dict[str, object]) -> dict[str, float]:
    return {
        "dense_standardized_mean_shift": max(
            row["standardized_mean_shift"]
            for row in report["dense"].values()
        ),
        "sparse_unseen_bucket_rate": max(
            row["unseen_bucket_rate"]
            for row in report["sparse"].values()
        ),
    }


def run_learning_benchmark(
    config: LearningBenchmarkConfig,
    output: Path,
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("learning benchmark output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fixture = FullFlowFixtureConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        route_k=24,
        merged_k=config.merged_k,
        coarse_k=config.coarse_k,
        fine_k=config.fine_k,
        expose_k=config.expose_k,
        history_length=config.history_length,
        recall_negatives=config.recall_negatives,
        scenario="feed_posting_cycle",
    )
    started = perf_counter()
    snapshots = build_full_flow_fixtures(fixture, ticks=config.ticks)
    generation_seconds = perf_counter() - started
    dataset_root = output / "dataset"
    materialize_started = perf_counter()
    partitions = []
    for tick, snapshot in enumerate(snapshots):
        partitions.append(append_full_flow_partition(
            snapshot, dataset_root, f"event_time={tick}",
        ))
    materialize_seconds = perf_counter() - materialize_started
    dataset = verify_full_flow_dataset(dataset_root)
    bus = PartitionedSampleBus(dataset_root, output / "lane-state")
    active_refs = bus.poll(Lane.ACTIVE)
    candidate_refs = bus.poll(Lane.CANDIDATE)
    if active_refs != candidate_refs:
        raise AssertionError("active and candidate lanes saw different data")
    load_started = perf_counter()
    batch = load_probe_batch(bus, active_refs)
    load_seconds = perf_counter() - load_started
    first = load_probe_batch(bus, active_refs[:1])
    last = load_probe_batch(bus, active_refs[-1:])
    drift = feature_drift_report(first, last)
    train_started = perf_counter()
    active_artifact = train_probe(batch, lane=Lane.ACTIVE, device=device, seed=71)
    candidate_artifact = train_probe(
        batch, lane=Lane.CANDIDATE, device=device, seed=71,
    )
    training_seconds = perf_counter() - train_started
    corpus_sha256 = _catalog_hash(snapshots[-1].catalog)
    compatibility = bus.compatibility(
        index_version=snapshots[-1].trace.manifest.index_version,
        corpus_sha256=corpus_sha256,
    )
    registry = PersistentModelRegistry(output / "registry")
    active = registry.register_candidate(
        active_artifact,
        compatibility,
        lane=Lane.ACTIVE,
        data_watermark=batch.event_watermark,
    )
    registry.shadow(active.serving_version_id, validation_status="pass")
    registry.promote(active.serving_version_id)
    candidate = registry.register_candidate(
        candidate_artifact,
        compatibility,
        lane=Lane.CANDIDATE,
        data_watermark=batch.event_watermark,
    )
    registry.shadow(candidate.serving_version_id, validation_status="hold")
    loaded, _, fallback = registry.load_active_with_fallback(compatibility)
    for ref in active_refs:
        bus.commit(Lane.ACTIVE, ref)
        bus.commit(Lane.CANDIDATE, ref)
    records = registry.checkpoint_records(created_time=config.ticks)
    report = {
        "schema": "p3-streaming-feature-infrastructure-benchmark-v1",
        "quality_claim": False,
        "config": asdict(config),
        "dataset_content_sha256": dataset["dataset_content_sha256"],
        "dataset_contract_hash": bus.contract_hash,
        "feature_manifest_hash": batch.feature_manifest_hash,
        "corpus_sha256": corpus_sha256,
        "partitions": partitions,
        "rows": len(batch.request_id),
        "mature_labels": int(batch.label_mask.sum()),
        "dense_feature_width": batch.dense_features.shape[1],
        "sparse_feature_width": batch.sparse_buckets.shape[1],
        "feature_drift": _max_drift(drift),
        "active_lane_cursor": bus.cursor(Lane.ACTIVE).manifest(),
        "candidate_lane_cursor": bus.cursor(Lane.CANDIDATE).manifest(),
        "active_probe": active_artifact.training_report,
        "candidate_probe": candidate_artifact.training_report,
        "active_serving_version_id": loaded.serving_version_id,
        "candidate_serving_version_id": candidate.serving_version_id,
        "candidate_status": registry.record(candidate.serving_version_id).status,
        "fallback_used": fallback,
        "checkpoint_records": [asdict(record) for record in records],
        "timing_seconds": {
            "generation": generation_seconds,
            "materialization": materialize_seconds,
            "loading": load_seconds,
            "training": training_seconds,
            "total": perf_counter() - started,
        },
        "peak_cuda_gib": _peak_cuda_gib(device),
        "evidence_boundary": (
            "P3-04/05 streaming, feature parity and lifecycle infrastructure; "
            "the LR probe is not a recommendation model launch or A/B lift."
        ),
    }
    replace_json_atomic(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--items", type=int, default=100_000)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_learning_benchmark(LearningBenchmarkConfig(
        users=args.users,
        items=args.items,
        ticks=args.ticks,
        device=args.device,
    ), args.output)
    print((args.output / "report.json").read_text())


if __name__ == "__main__":
    main()
