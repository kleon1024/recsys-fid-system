from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin import (
    AtomicSimulationKernel,
    CascadePolicy,
    EventType,
    ExperimentPlan,
    ObservableEventLog,
    RankingConfig,
    ReferencePlatformConfig,
    ReferenceRecommendationPlatform,
    RetrievalConfig,
    Surface,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
)


def _loop():
    users, items = 128, 4_000
    catalog = build_public_catalog(
        items=items,
        creators=256,
        merchants=80,
        advertisers=40,
        topics=16,
        countries=4,
        regions_per_country=4,
        embedding_dim=12,
        platform_seed=4_001,
        device="cpu",
        initial_active_fraction=0.75,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=users,
        topics=16,
        embedding_dim=12,
        countries=4,
        regions_per_country=4,
        environment_seed=4_003,
        future_signup_fraction=0.0,
    ), catalog)
    platform = ReferenceRecommendationPlatform(
        ReferencePlatformConfig(users=users, history_length=16),
        catalog,
        RetrievalConfig(route_k=24, merged_k=48, graph_neighbors=12),
        RankingConfig(coarse_k=32, fine_k=16, expose_k=8),
    )
    log = ObservableEventLog(allowed_lateness=world.max_reporting_lag)
    policy = CascadePolicy(
        "cold-start-loop",
        1,
        1,
        1,
        enabled_routes=("posting_context", "cold_start"),
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=policy,
        treatment_policy=policy,
        experiment_seed=4_009,
        control_fraction=0.5,
        treatment_fraction=0.5,
    )
    return world, platform, AtomicSimulationKernel(world, platform, log), plan


def _force_surface(world: UserEcosystemWorld, surface: Surface) -> None:
    world.users.active.fill_(True)
    world.users.surface_intent.fill_(1e-8)
    world.users.surface_intent[:, int(surface)] = 1.0


def test_posting_creates_the_exact_post_consumed_by_next_feed():
    world, platform, kernel, plan = _loop()
    _force_surface(world, Surface.POSTING)
    world.users.habit.fill_(1.0)
    posting = kernel.step(0, plan)
    assert torch.equal(
        platform.projection.state.user_creator_id,
        world.users.creator_id,
    )
    published = posting.response_events.event(EventType.PUBLISH)
    assert int(published.sum()) > 0
    posts = posting.response_events.post_id[published]
    sources = posting.response_events.source_candidate_id[published]
    assert torch.unique(posts).numel() == len(posts)
    assert (posts != sources).all()
    assert platform.projection.state.item_active[posts].all()
    assert (
        platform.projection.state.item_publish_time[posts] == 0
    ).all()
    creator = posting.response_events.creator_id[published]
    motivation_before_feed = world.supply.state.creator_motivation.clone()

    _force_surface(world, Surface.FEED)
    feed = kernel.step(1, plan)
    assert torch.isin(feed.candidate_trace.route_item_id, posts).any()
    impression = feed.response_events.event(EventType.IMPRESSION)
    impressed_posts = feed.response_events.item_id[
        impression & torch.isin(feed.response_events.item_id, posts)
    ]
    assert len(impressed_posts) > 0
    impressed_creator = platform.projection.state.item_creator_id[
        impressed_posts
    ]
    assert torch.isin(impressed_creator, creator).all()
    assert (
        world.supply.state.creator_motivation[impressed_creator]
        != motivation_before_feed[impressed_creator]
    ).any()
    assert (world.users.creation_inspiration_strength > 0.0).any()
    inspired = world.users.creation_inspiration_strength.clone()
    _force_surface(world, Surface.POSTING)
    inspired_posting = kernel.step(2, plan)
    published_again = inspired_posting.response_events.event(EventType.PUBLISH)
    assert published_again.any()
    published_user = inspired_posting.response_events.user_id[published_again]
    assert (
        world.users.creation_inspiration_strength[published_user]
        < inspired[published_user]
    ).any()
