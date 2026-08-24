from __future__ import annotations

import ast
from pathlib import Path

import torch

from fid_lab.simulation.digital_twin import (
    AtomicSimulationKernel,
    CascadePolicy,
    ExperimentPlan,
    JoinerConfig,
    ObservableEventLog,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RequestLevelJoiner,
    RetrievalConfig,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
)
from fid_lab.simulation.digital_twin.platform.retrieval import (
    ROUTE_NAMES,
    surface_eligibility,
)


def build_system(users=256, items=2_400):
    catalog = build_public_catalog(
        items=items,
        creators=120,
        merchants=60,
        advertisers=30,
        topics=16,
        countries=4,
        regions_per_country=6,
        embedding_dim=12,
        platform_seed=601,
        device="cpu",
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=users,
        topics=16,
        embedding_dim=12,
        countries=4,
        regions_per_country=6,
        environment_seed=607,
        future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=users, history_length=16),
        catalog,
        RetrievalConfig(
            route_k=8,
            merged_k=32,
            graph_neighbors=8,
            refresh_interval=1,
        ),
        RankingConfig(coarse_k=16, fine_k=8, expose_k=4),
    )
    log = ObservableEventLog(allowed_lateness=world.max_reporting_lag)
    kernel = AtomicSimulationKernel(world, platform, log)
    control = CascadePolicy("rules-v1", 1, 1, 1)
    treatment = CascadePolicy(
        "cross-sequence-v2",
        2,
        2,
        2,
        cross_weight=0.20,
        sequence_weight=0.25,
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=control,
        treatment_policy=treatment,
        experiment_seed=613,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    return world, platform, log, kernel, plan, catalog


def test_reference_cascade_emits_closed_stage_trace_for_partial_ab():
    _, _, _, kernel, plan, catalog = build_system()
    result = kernel.step(0, plan)
    trace = result.candidate_trace
    assert trace is not None and result.request_context is not None
    assert trace.recall_item_id.shape == (256, 32)
    assert trace.coarse_item_id.shape == (256, 16)
    assert trace.fine_item_id.shape == (256, 8)
    assert trace.exposed_item_id.shape == (256, 4)
    assert set(trace.fine_version_id.tolist()) == {1, 2}
    assert result.baseline_requests > result.experiment_requests
    safe = trace.recall_item_id.clamp_min(0)
    eligible = surface_eligibility(
        trace.surface,
        catalog.content_kind[safe],
    )
    assert (eligible | (trace.recall_item_id < 0)).all()
    assert (
        result.request_context.feature_as_of_ingest_time
        == trace.event_time
    ).all()


def test_routes_gain_behavioral_graph_and_keep_search_triggered():
    _, _, _, kernel, plan, _ = build_system()
    first = kernel.step(0, plan)
    second = kernel.step(1, plan)
    ann_bit = 1 << ROUTE_NAMES.index("ann")
    popular_bit = 1 << ROUTE_NAMES.index("popular")
    graph_bit = 1 << ROUTE_NAMES.index("graph")
    search_bit = 1 << ROUTE_NAMES.index("search")
    assert (first.candidate_trace.recall_route_id & ann_bit).any()
    assert (first.candidate_trace.recall_route_id & popular_bit).any()
    assert (second.candidate_trace.recall_route_id & graph_bit).any()
    search_rows = second.candidate_trace.surface == 1
    has_search_route = (
        second.candidate_trace.recall_route_id & search_bit
    ).any(dim=1)
    assert not has_search_route[~search_rows].any()
    if search_rows.any():
        assert has_search_route[search_rows].any()


def test_policy_can_change_only_retrieval_routes_and_version():
    world, platform, log, _, _, _ = build_system(users=192, items=2_000)
    kernel = AtomicSimulationKernel(world, platform, log)
    base = CascadePolicy(
        "base-routes", 1, 1, 1,
        recall_version_id=10,
        enabled_routes=("popular", "geo", "graph"),
    )
    treatment = CascadePolicy(
        "add-fresh", 1, 1, 1,
        recall_version_id=11,
        enabled_routes=("popular", "geo", "graph", "fresh"),
    )
    result = kernel.step(0, ExperimentPlan.ramped_user_ab(
        active_policy=base,
        treatment_policy=treatment,
        experiment_seed=97,
        control_fraction=0.4,
        treatment_fraction=0.4,
    ))
    trace = result.candidate_trace
    assert trace is not None
    control = trace.experiment_cell == 0
    treated = trace.experiment_cell == 1
    assert (trace.recall_version_id[control] == 10).all()
    assert (trace.recall_version_id[treated] == 11).all()
    fresh_bit = 1 << ROUTE_NAMES.index("fresh")
    assert not (trace.recall_route_id[control] & fresh_bit).any()
    assert (trace.recall_route_id[treated] & fresh_bit).any()


def test_real_trace_materializes_three_authorities_without_fake_negatives():
    _, _, _, kernel, plan, catalog = build_system()
    result = kernel.step(0, plan)
    joined = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=6), catalog,
    ).materialize(
        result.candidate_trace,
        result.request_context,
        result.response_events,
        event_watermark=0,
    )
    assert len(joined.recall.request_id) > 0
    assert joined.fine.item_id.shape == (256, 4)
    assert joined.coarse.item_id.shape == (256, 16)
    unexposed = ~joined.coarse.hard_label_mask
    assert (joined.coarse.hard_label[unexposed] == 0).all()
    assert torch.equal(
        joined.coarse.teacher_mask.sum(dim=1),
        (result.candidate_trace.fine_item_id >= 0).sum(dim=1),
    )


def test_platform_package_cannot_import_hidden_world_modules():
    for path in Path("fid_lab/simulation/digital_twin/platform").glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(module and "world" in module for module in imports), path
