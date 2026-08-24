from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin import (
    EventType,
    RenderedSlateBatch,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
    make_app_events,
)
from fid_lab.feed_loop.world_model.contracts import WorldModelConfig
from fid_lab.feed_loop.world_model.ensemble import WorldModelEnsemble
from fid_lab.simulation.digital_twin.world.authority import (
    NeuralFeedResponseAuthority,
)
from fid_lab.simulation.digital_twin.world.dynamics.population import (
    sample_population,
)
from fid_lab.simulation.digital_twin.world.dynamics.trends import TrendProcess


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    matrix = torch.corrcoef(torch.stack((left.float(), right.float())))
    return float(matrix[0, 1])


def test_population_is_deterministic_correlated_and_heterogeneous():
    users = torch.arange(20_000)
    left = sample_population(
        users, topics=64, countries=12, regions_per_country=16, seed=103,
    )
    right = sample_population(
        users, topics=64, countries=12, regions_per_country=16, seed=103,
    )
    for name in left.__dataclass_fields__:
        torch.testing.assert_close(getattr(left, name), getattr(right, name))
    assert torch.unique(left.mixture).numel() == 6
    assert torch.unique(left.country).numel() == 12
    assert torch.unique(left.lifecycle_cohort).numel() == 3
    assert torch.allclose(left.weekly_activity.mean(1), torch.ones(len(users)))
    assert _correlation(left.habit, left.activity) > 0.35
    assert _correlation(left.spending_power, left.activity) < 0.45
    assert _correlation(left.satisfaction, left.fatigue) < 0.35
    assert torch.allclose(left.surface_intent.sum(1), torch.ones(len(users)))


def test_trend_process_has_exogenous_and_factual_endogenous_components():
    left = TrendProcess(regions=4, topics=8, seed=103, device="cpu")
    right = TrendProcess(regions=4, topics=8, seed=103, device="cpu")
    left.advance(7)
    right.advance(7)
    torch.testing.assert_close(left.snapshot(), right.snapshot())
    event = make_app_events(
        EventType.SHARE,
        event_time=7,
        request_id=torch.tensor([11]),
        user_id=torch.tensor([3]),
        surface=torch.tensor([0]),
        item_id=torch.tensor([5]),
        region=torch.tensor([2]),
        topic_id=torch.tensor([6]),
    )
    left.commit(event)
    left.advance(8)
    right.advance(8)
    assert left.snapshot()[2, 6] > right.snapshot()[2, 6]


def test_session_survival_can_create_churn_without_observable_churn_label():
    catalog = build_public_catalog(
        items=200,
        creators=40,
        merchants=20,
        topics=8,
        countries=4,
        regions_per_country=3,
        embedding_dim=8,
        platform_seed=101,
        device="cpu",
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=2_000,
        topics=8,
        embedding_dim=8,
        countries=4,
        regions_per_country=3,
        environment_seed=103,
        future_signup_fraction=0.0,
    ), catalog)
    users = world.users
    users.active.fill_(True)
    users.satisfaction.zero_()
    users.fatigue.fill_(1.0)
    users.habit.fill_(0.01)
    users.churn_susceptibility.fill_(1.0)
    users.session_count.fill_(1)
    event = make_app_events(
        EventType.SESSION_END,
        event_time=10,
        request_id=users.user_id + 1,
        user_id=users.user_id,
        surface=torch.zeros_like(users.user_id),
    )
    world.commit(event)
    churn_rate = float(users.churned.float().mean())
    assert 0.65 < churn_rate < 0.85
    assert not hasattr(event, "churned")


def test_neural_feed_authority_is_request_keyed_and_keeps_hidden_inputs_private():
    catalog = build_public_catalog(
        items=240,
        creators=40,
        merchants=20,
        topics=8,
        countries=4,
        regions_per_country=3,
        embedding_dim=8,
        platform_seed=101,
        device="cpu",
    )
    ensemble = WorldModelEnsemble(WorldModelConfig(
        width=32,
        latent_dim=8,
        attention_heads=4,
        ensemble_members=2,
        batch_size=8,
        epochs=1,
    ))
    authority = NeuralFeedResponseAuthority(
        ensemble,
        member_index=0,
        artifact_sha256="a" * 64,
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=8,
        topics=8,
        embedding_dim=8,
        countries=4,
        regions_per_country=3,
        environment_seed=103,
        future_signup_fraction=0.0,
    ), catalog, response_authority=authority)
    user = torch.arange(8)
    position = torch.arange(5)[None].expand(8, -1)
    item = torch.remainder(user[:, None] * 17 + position * 29, 240)
    slate = RenderedSlateBatch(
        request_id=user + 1,
        user_id=user,
        surface=torch.zeros_like(user),
        event_time=torch.zeros_like(user),
        item_ids=item,
        positions=position,
        valid=torch.ones_like(item, dtype=torch.bool),
        ui_variant=torch.zeros_like(user),
        exposure_probability=torch.ones_like(item, dtype=torch.float),
        assignment_probability=torch.ones_like(user, dtype=torch.float),
    )
    snapshot = world.snapshot()
    first = authority.respond(snapshot, catalog, slate, 103)
    second = authority.respond(snapshot, catalog, slate, 103)
    assert torch.equal(first.event_id, second.event_id)
    assert torch.equal(first.event_type, second.event_type)
    assert first.event(EventType.IMPRESSION).sum() == 40
    assert "neural-feed" in world.manifest()["response"]
    assert not hasattr(slate, "selected_features")
