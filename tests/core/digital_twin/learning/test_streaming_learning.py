from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fid_lab.simulation.digital_twin.learning import (
    Lane,
    PartitionedSampleBus,
    PersistentModelRegistry,
    feature_drift_report,
    load_probe_batch,
    train_probe,
)
from fid_lab.simulation.digital_twin.learning.contracts import (
    ServingCompatibility,
    content_hash,
)
from fid_lab.simulation.digital_twin.learning.benchmark import (
    LearningBenchmarkConfig,
    run_learning_benchmark,
)
from fid_lab.simulation.digital_twin.platform.features import FeatureTensorBatch
from fid_lab.simulation.digital_twin.observability import (
    FullFlowFixtureConfig,
    append_full_flow_partition,
    build_full_flow_fixtures,
)


def build_dataset(root: Path) -> None:
    snapshots = build_full_flow_fixtures(FullFlowFixtureConfig(
        users=192,
        items=3_000,
        scenario="feed_posting_cycle",
    ), ticks=2)
    for tick, snapshot in enumerate(snapshots):
        append_full_flow_partition(snapshot, root, f"event_time={tick}")


def test_active_and_candidate_lanes_share_data_but_resume_independently(tmp_path):
    dataset = tmp_path / "dataset"
    build_dataset(dataset)
    bus = PartitionedSampleBus(dataset, tmp_path / "lane-state")
    active = bus.poll(Lane.ACTIVE)
    candidate = bus.poll(Lane.CANDIDATE)
    assert active == candidate and len(active) == 2
    with pytest.raises(ValueError, match="event-time order"):
        bus.commit(Lane.ACTIVE, active[1])
    committed = bus.commit(Lane.ACTIVE, active[0])
    assert committed["status"] == "committed"
    assert bus.commit(Lane.ACTIVE, active[0])["status"] == "resumed"
    assert bus.poll(Lane.ACTIVE) == (active[1],)
    assert bus.poll(Lane.CANDIDATE) == candidate
    bus.commit(Lane.CANDIDATE, candidate[0])
    restarted = PartitionedSampleBus(dataset, tmp_path / "lane-state")
    assert restarted.cursor(Lane.ACTIVE).consumed == (
        (active[0].key, active[0].content_sha256),
    )
    assert restarted.cursor(Lane.CANDIDATE).consumed == (
        (candidate[0].key, candidate[0].content_sha256),
    )


def test_probe_registry_rejects_mismatch_and_preserves_fallback(tmp_path):
    dataset = tmp_path / "dataset"
    build_dataset(dataset)
    bus = PartitionedSampleBus(dataset, tmp_path / "lane-state")
    refs = bus.poll(Lane.ACTIVE)
    batch = load_probe_batch(bus, refs)
    assert batch.position.shape == batch.item_id.shape
    assert batch.exposed.shape == batch.item_id.shape
    assert batch.randomized_support.shape == batch.item_id.shape
    assert torch.equal(
        batch.label_mask, batch.label_applicable & batch.label_mature,
    )
    no_drift = feature_drift_report(batch, batch)
    assert max(
        row["standardized_mean_shift"]
        for row in no_drift["dense"].values()
    ) == 0.0
    shifted = replace(
        batch,
        dense_features=batch.dense_features + torch.tensor(
            [1.0] + [0.0] * (batch.dense_features.shape[1] - 1),
        ),
    )
    drift = feature_drift_report(batch, shifted)
    assert drift["dense"][batch.dense_feature_names[0]][
        "standardized_mean_shift"
    ] > 0.0
    active_probe = train_probe(batch, lane=Lane.ACTIVE, seed=41)
    candidate_probe = train_probe(batch, lane=Lane.CANDIDATE, seed=41)
    for name, value in active_probe.model.state_dict().items():
        assert torch.equal(value, candidate_probe.model.state_dict()[name])
    assert active_probe.training_report["purpose"] == (
        "infrastructure_only_not_model_launch"
    )
    compatibility = bus.compatibility(
        index_version="observable-index-t1",
        corpus_sha256=content_hash({"catalog": "public-catalog-v1"}),
    )
    registry = PersistentModelRegistry(tmp_path / "registry")
    first = registry.register_candidate(
        active_probe,
        compatibility,
        lane=Lane.ACTIVE,
        data_watermark=batch.event_watermark,
    )
    registry.shadow(first.serving_version_id, validation_status="pass")
    registry.promote(first.serving_version_id)
    restarted = PersistentModelRegistry(tmp_path / "registry")
    loaded, artifact = restarted.load("active", compatibility)
    assert loaded.serving_version_id == first.serving_version_id
    assert artifact.feature_manifest_hash == batch.feature_manifest_hash
    replay_score = artifact.score(FeatureTensorBatch(
        dense=batch.dense_features[:8, None, :],
        sparse_fids=torch.zeros(
            8, 1, batch.sparse_buckets.shape[1], dtype=torch.long,
        ),
        sparse_buckets=batch.sparse_buckets[:8, None, :],
        manifest_hash=batch.feature_manifest_hash,
    ), batch.surface[:8])
    assert replay_score.shape == (8, 1)
    assert torch.isfinite(replay_score).all()
    serving = ServingCompatibility(
        feature_manifest_hash=compatibility.feature_manifest_hash,
        feature_version=compatibility.feature_version,
        fid_version=compatibility.fid_version,
        catalog_version=compatibility.catalog_version,
        index_version=compatibility.index_version,
        code_sha256=compatibility.code_sha256,
    )
    served, _ = restarted.load_version_for_serving(
        first.serving_version_id, serving,
    )
    assert served.serving_version_id == first.serving_version_id
    with pytest.raises(ValueError, match="serving index_version"):
        restarted.load_version_for_serving(
            first.serving_version_id,
            replace(serving, index_version="wrong-index"),
        )
    with pytest.raises(ValueError, match="incompatible"):
        restarted.load(
            "active", replace(compatibility, index_version="wrong-index"),
        )

    second = restarted.register_candidate(
        candidate_probe,
        compatibility,
        lane=Lane.CANDIDATE,
        data_watermark=batch.event_watermark,
    )
    restarted.shadow(second.serving_version_id, validation_status="pass")
    restarted.promote(second.serving_version_id)
    assert restarted.alias("fallback").serving_version_id == (
        first.serving_version_id
    )
    active_path = restarted.artifacts / restarted.alias("active").artifact_file
    active_path.write_bytes(active_path.read_bytes() + b"corrupt")
    fallback, _, used_fallback = restarted.load_active_with_fallback(
        compatibility,
    )
    assert used_fallback
    assert fallback.serving_version_id == first.serving_version_id

    third = restarted.register_candidate(
        active_probe,
        compatibility,
        lane=Lane.CANDIDATE,
        data_watermark=batch.event_watermark,
    )
    restarted.reject(third.serving_version_id)
    assert restarted.alias("active").serving_version_id == (
        second.serving_version_id
    )
    rows = restarted.checkpoint_records(created_time=2)
    assert {row.serving_version_id for row in rows} == {
        first.serving_version_id,
        second.serving_version_id,
        third.serving_version_id,
    }
    assert all(row.compatibility_hash for row in rows)


def test_learning_benchmark_keeps_probe_out_of_launch_claims(tmp_path):
    report = run_learning_benchmark(
        LearningBenchmarkConfig(
            users=96, items=1_500, ticks=2, device="cpu",
        ),
        tmp_path / "benchmark",
    )
    assert not report["quality_claim"]
    assert report["candidate_status"] == "shadow"
    assert report["active_lane_cursor"]["consumed"] == (
        report["candidate_lane_cursor"]["consumed"]
    )
    assert report["dense_feature_width"] == 11
    assert report["sparse_feature_width"] == 13
