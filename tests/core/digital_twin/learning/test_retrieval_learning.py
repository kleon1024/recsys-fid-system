from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin.learning import (
    Lane,
    PartitionedSampleBus,
    PersistentModelRegistry,
)
from fid_lab.simulation.digital_twin.learning.retrieval import (
    RetrievalANNIndex,
    RetrievalArtifact,
    RetrievalModelConfig,
    corpus_from_snapshot,
    load_retrieval_batch,
    train_retrieval_model,
)
from fid_lab.simulation.digital_twin.learning.retrieval.benchmark import (
    RetrievalBenchmarkConfig,
    run_retrieval_benchmark,
)
from fid_lab.simulation.digital_twin.observability import (
    FullFlowFixtureConfig,
    append_full_flow_partition,
    build_full_flow_fixtures,
)


def _dataset(tmp_path):
    snapshots = build_full_flow_fixtures(FullFlowFixtureConfig(
        users=128,
        items=2_400,
        route_k=8,
        merged_k=32,
        coarse_k=16,
        fine_k=8,
        expose_k=4,
        recall_negatives=8,
        scenario="feed_consumption",
    ), ticks=3)
    root = tmp_path / "dataset"
    for tick, snapshot in enumerate(snapshots):
        append_full_flow_partition(snapshot, root, f"event_time={tick}")
    bus = PartitionedSampleBus(root, tmp_path / "state")
    return snapshots, bus, bus.poll(Lane.CANDIDATE)


def test_retrieval_loader_preserves_expected_count_and_pit_context(tmp_path):
    snapshots, bus, refs = _dataset(tmp_path)
    batch = load_retrieval_batch(bus, refs[:2])
    assert len(batch.request_id) > 0
    assert batch.negative_item_id.shape[1] == 8
    assert (batch.negative_expected_count[batch.negative_loss_mask] > 0).all()
    assert batch.history_item_id.shape == batch.history_event_type.shape
    assert batch.feature_manifest_hash == snapshots[-1].feature_manifest.manifest_hash


def test_retrieval_artifact_registry_round_trip_and_ann_budget(tmp_path):
    snapshots, bus, refs = _dataset(tmp_path)
    batch = load_retrieval_batch(bus, refs[:2])
    corpus = corpus_from_snapshot(snapshots[-1])
    config = RetrievalModelConfig(
        architecture="multi_interest",
        representation_dim=8,
        hidden_dim=16,
        embedding_dim=4,
        user_hash_buckets=512,
        item_hash_buckets=1_024,
        creator_hash_buckets=512,
        epochs=1,
        batch_size=64,
        seed=17,
    )
    model, training = train_retrieval_model(batch, corpus, config, device="cpu")
    artifact = RetrievalArtifact(
        model,
        config,
        batch.feature_manifest_hash,
        "test-retrieval-stage-contract",
        corpus.content_sha256,
        training,
    )
    compatibility = bus.compatibility(
        index_version=artifact.index_version,
        corpus_sha256=corpus.content_sha256,
        stage_contract_hash=artifact.retrieval_feature_contract_hash,
    )
    registry = PersistentModelRegistry(tmp_path / "registry")
    record = registry.register_candidate(
        artifact, compatibility, lane=Lane.CANDIDATE,
        data_watermark=batch.event_watermark,
    )
    loaded_record, loaded = registry.load(
        "candidate", compatibility, corpus=corpus,
    )
    assert loaded_record.serving_version_id == record.serving_version_id
    assert loaded.state_sha256 == artifact.state_sha256
    index = RetrievalANNIndex(loaded, corpus, device="cpu")
    query = loaded.model.encode_queries(batch.select(torch.arange(4)))
    item, score = index.search(query, 10)
    assert item.shape == score.shape == (4, 10)
    assert all(len(torch.unique(row)) == 10 for row in item)


def test_fixed_budget_benchmark_uses_one_corpus_and_emits_decisions(tmp_path):
    report = run_retrieval_benchmark(RetrievalBenchmarkConfig(
        users=128,
        items=2_400,
        ticks=3,
        device="cpu",
        top_k=10,
        downstream_k=5,
        route_k=8,
        merged_k=32,
        coarse_k=16,
        fine_k=8,
        expose_k=4,
        recall_negatives=8,
        epochs=1,
        batch_size=64,
        max_evaluation_queries=64,
        latency_budget_ms=100.0,
    ), tmp_path / "benchmark")
    assert report["same_corpus_top_k_latency_budget"] is True
    assert set(report["models"]) == {"two_tower", "multi_interest"}
    assert set(report["baseline"]) == {
        "lifecycle_rules", "co_visit_graph", "baseline_rrf",
    }
    assert all(
        row["decision"] in {"pass", "hold", "reject"}
        for row in report["models"].values()
    )
    assert report["candidate_lane_cursor"]["event_watermark"] == 2
