from __future__ import annotations

import pytest

from fid_lab.simulation.digital_twin import (
    AtomicSimulationKernel,
    CascadePolicy,
    ExperimentPlan,
    ObservableEventLog,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
    UserEcosystemWorld,
    UserWorldConfig,
    WorldBranchRegistry,
    WorldCheckpointStore,
    build_public_catalog,
)
from fid_lab.simulation.digital_twin.learning import FactualRequestStream


def _system():
    catalog = build_public_catalog(
        items=1_200,
        creators=80,
        merchants=40,
        advertisers=20,
        topics=12,
        countries=3,
        regions_per_country=4,
        embedding_dim=10,
        platform_seed=2_101,
        device="cpu",
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=128,
        topics=12,
        embedding_dim=10,
        countries=3,
        regions_per_country=4,
        environment_seed=2_103,
        future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=128, history_length=16),
        catalog,
        RetrievalConfig(route_k=8, merged_k=24, graph_neighbors=8),
        RankingConfig(coarse_k=12, fine_k=8, expose_k=4),
    )
    kernel = AtomicSimulationKernel(
        world,
        platform,
        ObservableEventLog(allowed_lateness=world.max_reporting_lag),
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=CascadePolicy("request-stream-active", 1, 1, 1),
        treatment_policy=CascadePolicy("request-stream-candidate", 2, 2, 2),
        experiment_seed=2_107,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    return kernel, plan


def _main_branch(tmp_path):
    kernel, plan = _system()
    tick = kernel.step(0, plan)
    checkpoints = WorldCheckpointStore(tmp_path / "checkpoints")
    checkpoint = checkpoints.save(kernel, 0, plan)
    branch = WorldBranchRegistry(checkpoints).initialize_main(
        checkpoint.checkpoint_id,
    )
    return kernel, tick, branch


def test_request_stream_is_idempotent_content_verified_and_resumable(tmp_path):
    kernel, tick, branch = _main_branch(tmp_path)
    stream = FactualRequestStream(tmp_path / "requests", branch)
    first = stream.append(
        tick, kernel.platform.projection.snapshot(), kernel.world.manifest(),
    )
    resumed = stream.append(
        tick, kernel.platform.projection.snapshot(), kernel.world.manifest(),
    )

    assert resumed == first
    assert stream.refs(training=True) == (first,)
    restored = FactualRequestStream(tmp_path / "requests", branch).read(first)
    assert restored.logical_time == 0
    assert restored.trace.request_id.equal(tick.candidate_trace.request_id)
    assert restored.context.request_id.equal(tick.request_context.request_id)
    assert restored.world_manifest == kernel.world.manifest()
    assert first.world_manifest_sha256
    assert len(restored.events.event_id) == (
        len(tick.entry_events.event_id) + len(tick.response_events.event_id)
    )
    moved = stream.read(first, device="cpu")
    assert moved.trace.request_id.device.type == "cpu"


def test_request_stream_stages_atomically_and_reconciles_orphans(tmp_path):
    kernel, tick, branch = _main_branch(tmp_path)
    stream = FactualRequestStream(tmp_path / "requests", branch)
    ref = stream.stage(
        "launch-attempt-1",
        tick,
        kernel.platform.projection.snapshot(),
        kernel.world.manifest(),
    )
    assert stream.refs(training=True) == ()
    stream.commit_staged("launch-attempt-1", (ref,))
    assert stream.refs(training=True) == (ref,)
    assert stream.reconcile_through(-1) == (ref,)
    assert stream.refs(training=True) == ()


def test_diagnostic_request_stream_cannot_be_used_for_training(tmp_path):
    kernel, tick, main = _main_branch(tmp_path)
    registry = WorldBranchRegistry(WorldCheckpointStore(tmp_path / "checkpoints"))
    shadow = registry.fork(
        main.name,
        "shadow/request-replay",
        kind="shadow",
        purpose="request trace replay",
    )
    stream = FactualRequestStream(tmp_path / "shadow-requests", shadow)
    stream.append(
        tick, kernel.platform.projection.snapshot(), kernel.world.manifest(),
    )

    assert len(stream.refs()) == 1
    with pytest.raises(ValueError, match="cannot train"):
        stream.refs(training=True)


def test_request_stream_detects_object_corruption(tmp_path):
    kernel, tick, branch = _main_branch(tmp_path)
    stream = FactualRequestStream(tmp_path / "requests", branch)
    ref = stream.append(
        tick, kernel.platform.projection.snapshot(), kernel.world.manifest(),
    )
    (stream.objects / f"{ref.object_sha256}.pt.zst").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="missing or corrupted"):
        stream.read(ref)
