from __future__ import annotations

import ast
from pathlib import Path

import torch

from fid_lab.simulation.digital_twin import (
    AtomicSimulationKernel,
    CascadePolicy,
    EventType,
    ExperimentPlan,
    JoinerConfig,
    ObservableEventLog,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RequestLevelJoiner,
    RetrievalConfig,
    SelectionPolicyKind,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
)
from fid_lab.simulation.digital_twin.platform.retrieval import (
    ROUTE_NAMES,
    surface_eligibility,
)
from fid_lab.simulation.digital_twin.platform.features import (
    DEFAULT_FEATURE_MANIFEST,
    FeatureTensorBatch,
)
from fid_lab.simulation.digital_twin.contracts import make_app_events


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
    assert trace.candidate_dense_features.shape == (256, 32, 11)
    assert trace.candidate_sparse_fids.shape == (256, 32, 13)
    assert trace.candidate_sparse_buckets.shape == (256, 32, 13)
    assert trace.manifest.feature_manifest_hash == (
        DEFAULT_FEATURE_MANIFEST.manifest_hash
    )
    valid_recall = trace.recall_item_id >= 0
    assert (trace.candidate_sparse_fids[valid_recall] > 0).all()
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
    report = kernel.platform.ranker.features.collision_report(
        kernel.platform.open_requests(result.entry_events),
        kernel.platform.snapshot().projection.state,
        trace.fine_item_id,
        kernel.platform.ranker._map_score(
            trace.fine_item_id,
            trace.recall_item_id,
            trace.recall_route_id,
        ),
    )
    assert report["feature_manifest_hash"] == trace.manifest.feature_manifest_hash
    assert all(
        field["fid_collisions"] == 0
        for field in report["fields"].values()
    )


def test_routes_gain_behavioral_graph_and_keep_search_triggered():
    _, _, _, kernel, plan, _ = build_system()
    first = kernel.step(0, plan)
    second = kernel.step(1, plan)
    ann_bit = 1 << ROUTE_NAMES.index("recent_ann")
    evergreen_bit = 1 << ROUTE_NAMES.index("evergreen")
    graph_bit = 1 << ROUTE_NAMES.index("recent_graph")
    search_bit = 1 << ROUTE_NAMES.index("search")
    assert (first.candidate_trace.recall_route_id & ann_bit).any()
    assert (first.candidate_trace.recall_route_id & evergreen_bit).any()
    assert (second.candidate_trace.recall_route_id & graph_bit).any()
    search_rows = second.candidate_trace.surface == 1
    has_search_route = (
        second.candidate_trace.recall_route_id & search_bit
    ).any(dim=1)
    assert not has_search_route[~search_rows].any()
    if search_rows.any():
        assert has_search_route[search_rows].any()


def test_final_ranker_pads_after_candidate_exhaustion_without_reexposure():
    _, platform, _, _, _, _ = build_system()
    item = torch.tensor([[10, 11, -1, -1, -1, -1, -1, -1]])
    score = torch.tensor([[0.8, 0.7, -torch.inf, -torch.inf,
                           -torch.inf, -torch.inf, -torch.inf, -torch.inf]])
    selected, selected_score = platform.ranker._diversified_top(item, score)
    assert selected.tolist() == [[10, 11, -1, -1]]
    assert torch.isfinite(selected_score[:, :2]).all()
    assert torch.isneginf(selected_score[:, 2:]).all()


def test_policy_can_change_only_retrieval_routes_and_version():
    world, platform, log, _, _, _ = build_system(users=192, items=2_000)
    kernel = AtomicSimulationKernel(world, platform, log)
    base = CascadePolicy(
        "base-routes", 1, 1, 1,
        recall_version_id=10,
        enabled_routes=("evergreen", "recent_ann", "recent_graph"),
    )
    treatment = CascadePolicy(
        "add-cold-start", 1, 1, 1,
        recall_version_id=11,
        enabled_routes=(
            "evergreen", "recent_ann", "recent_graph", "cold_start",
        ),
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
    cold_start_bit = 1 << ROUTE_NAMES.index("cold_start")
    assert not (trace.recall_route_id[control] & cold_start_bit).any()
    assert (trace.recall_route_id[treated] & cold_start_bit).any()


def test_randomized_cascade_logs_exact_support_without_changing_factuality():
    world, platform, log, _, _, _ = build_system(users=512, items=3_000)
    policy = CascadePolicy(
        "exploration-lane",
        1,
        1,
        1,
        exploration_rate=0.25,
        exploration_seed=113,
    )
    result = AtomicSimulationKernel(world, platform, log).step(
        0,
        ExperimentPlan.ramped_user_ab(
            active_policy=policy,
            treatment_policy=policy,
            experiment_seed=127,
            control_fraction=0.25,
            treatment_fraction=0.25,
        ),
    )
    trace = result.candidate_trace
    randomized = (
        trace.selection_policy_kind == int(SelectionPolicyKind.RANDOMIZED)
    )
    assert randomized.any() and (~randomized).any()
    assert (trace.exposure_probability[trace.exposed_item_id >= 0] > 0).all()
    assert (trace.coarse_admission_probability >= 0).all()
    assert (trace.fine_admission_probability >= 0).all()
    joined = RequestLevelJoiner(
        JoinerConfig(ticks_per_day=96, recall_negatives=6), platform.catalog,
    ).materialize(
        trace,
        result.request_context,
        result.response_events,
        event_watermark=0,
    )
    valid = joined.fine.item_id >= 0
    assert joined.fine.randomized_support[valid].all()
    assert (joined.fine.exposure_probability[~joined.fine.exposed] == 0).all()


def test_installed_learned_scorer_replays_exact_score_and_version():
    world, platform, log, _, _, _ = build_system(users=128, items=1_600)

    class AffinityScorer:
        feature_manifest_hash = DEFAULT_FEATURE_MANIFEST.manifest_hash

        @staticmethod
        def score(features, surface):
            del surface
            return features.dense[:, :, 0]

    serving_version = 77
    scorer = AffinityScorer()
    platform.install_fine_scorer(serving_version, scorer)
    policy = CascadePolicy("learned-probe", 1, serving_version, 1)
    result = AtomicSimulationKernel(world, platform, log).step(
        0,
        ExperimentPlan.ramped_user_ab(
            active_policy=policy,
            treatment_policy=policy,
            experiment_seed=91,
            control_fraction=0.25,
            treatment_fraction=0.25,
        ),
    )
    trace = result.candidate_trace
    assert trace is not None
    assert (trace.fine_version_id == serving_version).all()
    candidate_features = FeatureTensorBatch(
        dense=trace.candidate_dense_features,
        sparse_fids=trace.candidate_sparse_fids,
        sparse_buckets=trace.candidate_sparse_buckets,
        manifest_hash=trace.manifest.feature_manifest_hash,
    )
    scorer_input = platform.ranker._select_features(
        trace.coarse_item_id, trace.recall_item_id, candidate_features,
    )
    replay = scorer.score(scorer_input, trace.surface)
    valid = trace.coarse_item_id >= 0
    assert torch.allclose(replay[valid], trace.fine_input_score[valid])


def test_installed_learned_retriever_owns_ann_route_and_index_version():
    world, platform, log, _, plan, catalog = build_system(users=128, items=1_600)

    class FixedRetriever:
        serving_version_id = 81
        index_version = "learned-index-81"

        @staticmethod
        def retrieve(requests, state, top_k):
            eligible = state.item_active & surface_eligibility(
                0, catalog.content_kind,
            )
            items = catalog.item_id[eligible][:top_k]
            values = torch.linspace(1.0, 0.1, len(items))
            return (
                items[None].expand(len(requests.request_id), -1),
                values[None].expand(len(requests.request_id), -1),
            )

    platform.retriever.install_learned_retriever(FixedRetriever())
    result = AtomicSimulationKernel(world, platform, log).step(0, plan)
    trace = result.candidate_trace
    assert trace is not None
    assert trace.manifest.index_version == "learned-index-81"
    ann_bit = 1 << ROUTE_NAMES.index("recent_ann")
    assert (trace.recall_route_id & ann_bit).any()


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
    assert joined.fine.item_id.shape == (256, 16)
    assert joined.coarse.item_id.shape == (256, 32)
    unexposed = ~joined.coarse.hard_label_mask
    assert (joined.coarse.hard_label[unexposed] == 0).all()
    assert torch.equal(
        joined.coarse.teacher_mask.sum(dim=1),
        (result.candidate_trace.coarse_item_id >= 0).sum(dim=1),
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


def test_content_removal_updates_projection_and_rebuilds_ann_index():
    _, platform, _, kernel, plan, catalog = build_system()
    kernel.step(0, plan)
    post = torch.where(
        platform.projection.state.item_active
        & (catalog.content_kind <= 3)
    )[0][:1]
    removal = make_app_events(
        EventType.MODERATION_REMOVE,
        event_time=1,
        request_id=torch.tensor([9_001]),
        user_id=torch.tensor([-1]),
        surface=torch.tensor([-1]),
        item_id=post,
        post_id=post,
        creator_id=platform.projection.state.item_creator_id[post],
        content_kind=catalog.content_kind[post],
        topic_id=catalog.topic_id[post],
    )
    platform.ingest(removal)
    assert not platform.projection.state.item_active[post].any()
    assert not platform.retriever.faiss._indexed_active[post].any()
