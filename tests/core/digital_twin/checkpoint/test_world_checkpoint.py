from __future__ import annotations

from dataclasses import fields

import torch

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
    WorldCheckpointStore,
    build_public_catalog,
)


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
        RetrievalConfig(
            route_k=8,
            merged_k=24,
            graph_neighbors=8,
            refresh_interval=2,
        ),
        RankingConfig(coarse_k=12, fine_k=8, expose_k=4),
    )
    kernel = AtomicSimulationKernel(
        world,
        platform,
        ObservableEventLog(allowed_lateness=world.max_reporting_lag),
    )
    active = CascadePolicy("checkpoint-control", 1, 1, 1)
    treatment = CascadePolicy(
        "checkpoint-treatment",
        2,
        2,
        2,
        enabled_routes=("evergreen", "recent_ann", "recent_graph"),
        exploration_rate=0.05,
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=active,
        treatment_policy=treatment,
        experiment_seed=2_107,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    return kernel, plan


def _assert_tensor_dataclass_equal(left, right):
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, torch.Tensor):
            torch.testing.assert_close(left_value, right_value)
        else:
            assert left_value == right_value


def _assert_tick_equal(left, right):
    assert left.logical_time == right.logical_time
    assert left.cell_counts == right.cell_counts
    _assert_tensor_dataclass_equal(left.entry_events, right.entry_events)
    _assert_tensor_dataclass_equal(left.response_events, right.response_events)
    _assert_tensor_dataclass_equal(left.candidate_trace, right.candidate_trace)
    _assert_tensor_dataclass_equal(left.request_context, right.request_context)


def test_checkpoint_restore_and_fork_preserve_the_next_factual_tick(tmp_path):
    source, plan = _system()
    source.step(0, plan)
    source.step(1, plan)
    store = WorldCheckpointStore(tmp_path)
    ref = store.save(
        source,
        logical_time=1,
        experiment=plan,
        learning_cursors={"stream": {"partition": "event_time=1"}},
    )

    left, _ = _system()
    right, _ = _system()
    restored_left = store.restore(left, ref.checkpoint_id)
    restored_right = store.restore(right, ref.checkpoint_id)
    assert restored_left.learning_cursors == {
        "stream": {"partition": "event_time=1"},
    }
    assert restored_left.ref == restored_right.ref == ref

    expected = source.step(2, plan)
    left_tick = left.step(2, restored_left.experiment)
    right_tick = right.step(2, restored_right.experiment)
    _assert_tick_equal(expected, left_tick)
    _assert_tick_equal(left_tick, right_tick)
    _assert_tensor_dataclass_equal(source.world.users, left.world.users)
    _assert_tensor_dataclass_equal(source.world.supply.state, left.world.supply.state)
    _assert_tensor_dataclass_equal(
        source.platform.projection.state,
        left.platform.projection.state,
    )


def test_checkpoint_rejects_an_incompatible_catalog(tmp_path):
    source, plan = _system()
    source.step(0, plan)
    ref = WorldCheckpointStore(tmp_path).save(source, 0, plan)
    incompatible, _ = _system()
    incompatible.world.catalog.quality_prior[0] += 0.1
    try:
        WorldCheckpointStore(tmp_path).restore(incompatible, ref.checkpoint_id)
    except ValueError as error:
        assert "catalog" in str(error)
    else:
        raise AssertionError("catalog skew must fail checkpoint restore")
