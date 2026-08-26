from __future__ import annotations

from dataclasses import fields

import torch

from fid_lab.simulation.digital_twin import (
    EventType,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
    make_app_events,
)
from fid_lab.simulation.digital_twin.world.dynamics.calendar import (
    sample_return_outcome,
)
from fid_lab.simulation.digital_twin.world.dynamics.growth import (
    AcquisitionChannel,
    GrowthProcess,
)
from fid_lab.simulation.digital_twin.world.dynamics.needs import (
    NeedKind,
    refresh_expired_needs,
)
from fid_lab.simulation.digital_twin.world.state import HiddenUserState
from fid_lab.simulation.digital_twin.platform.projection import (
    PlatformProjectionState,
)


def _world(users=20_000):
    catalog = build_public_catalog(
        items=20_000,
        creators=1_000,
        merchants=200,
        topics=32,
        countries=6,
        regions_per_country=8,
        embedding_dim=16,
        platform_seed=701,
        device="cpu",
    )
    return UserEcosystemWorld(UserWorldConfig(
        users=users,
        topics=32,
        embedding_dim=16,
        countries=6,
        regions_per_country=8,
        environment_seed=709,
        ticks_per_day=96,
        future_signup_fraction=0.20,
    ), catalog)


def test_equilibrium_population_has_distinct_growth_need_and_lifecycle_states():
    world = _world()
    users = world.users
    assert torch.unique(users.acquisition_channel).numel() == len(
        AcquisitionChannel
    )
    assert torch.unique(users.need_kind).numel() == len(NeedKind)
    assert torch.unique(users.lifecycle_stage).numel() >= 3
    assert (users.registered & (users.signup_time < 0)).any()
    assert ((~users.registered) & (users.signup_time > 0)).any()
    assert users.need_strength.std() > 0.10
    assert users.activation_score.std() > 0.05
    observable = {field.name for field in fields(PlatformProjectionState)}
    assert not observable & {
        "acquisition_quality",
        "need_kind",
        "need_strength",
        "activation_score",
        "session_value_ema",
    }


def test_product_led_feedback_increases_only_referral_acquisition_pressure():
    process = GrowthProcess(countries=2, seed=719, device=torch.device("cpu"))
    process.advance(0, 96)
    channel = torch.full((128,), int(AcquisitionChannel.REFERRAL))
    quality = torch.full((128,), 0.6)
    susceptibility = torch.ones(128)
    country = torch.cat((torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)))
    before = process.registration_probability(
        channel, quality, susceptibility, country,
    )
    shares = make_app_events(
        EventType.SHARE,
        event_time=0,
        request_id=torch.arange(1, 65),
        user_id=torch.arange(64),
        surface=torch.zeros(64, dtype=torch.long),
        item_id=torch.arange(64),
        country=torch.zeros(64, dtype=torch.long),
    )
    process.commit(shares)
    after = process.registration_probability(
        channel, quality, susceptibility, country,
    )
    assert (after[:64] > before[:64]).all()
    torch.testing.assert_close(after[64:], before[64:])


def test_need_refresh_is_heterogeneous_and_counter_deterministic():
    users = torch.arange(5_000)
    arguments = {
        "user": users,
        "need_kind": torch.zeros_like(users),
        "need_topic": torch.zeros_like(users),
        "need_strength": torch.zeros(5_000),
        "need_expiry_time": torch.zeros_like(users),
        "primary_topic": users.remainder(32),
        "secondary_topic": (users + 7).remainder(32),
        "logical_time": 11,
        "ticks_per_day": 96,
        "topics": 32,
        "seed": 727,
    }
    left = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in arguments.items()}
    right = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in arguments.items()}
    refresh_expired_needs(**left)
    refresh_expired_needs(**right)
    for name in ("need_kind", "need_topic", "need_strength", "need_expiry_time"):
        torch.testing.assert_close(left[name], right[name])
    assert torch.unique(left["need_kind"]).numel() == len(NeedKind)
    assert torch.unique(left["need_topic"]).numel() == 32
    assert left["need_strength"].std() > 0.10


def test_activation_and_session_value_reduce_return_delay_and_churn():
    world = _world(users=4_000)
    state = world.users
    common = torch.arange(1_000)
    low_index = torch.arange(1_000)
    high_index = torch.arange(1_000, 2_000)
    for index in (low_index, high_index):
        state.satisfaction[index] = 0.55
        state.fatigue[index] = 0.35
        state.habit[index] = 0.45
        state.churn_susceptibility[index] = 0.50
        state.session_count[index] = 8
        state.need_strength[index] = 0.50
    state.activation_score[low_index] = 0.15
    state.acquisition_quality[low_index] = 0.25
    state.session_value_ema[low_index] = -0.5
    state.return_streak[low_index] = 0
    state.activation_score[high_index] = 0.90
    state.acquisition_quality[high_index] = 0.85
    state.session_value_ema[high_index] = 1.5
    state.return_streak[high_index] = 8

    def selected(index):
        return HiddenUserState(**{
            field.name: getattr(state, field.name)[index]
            for field in fields(HiddenUserState)
        })

    event_id = 50_000 + common
    event_time = torch.full_like(common, 96)
    low = sample_return_outcome(
        selected(low_index), event_id, event_time, 96, 733,
    )
    high = sample_return_outcome(
        selected(high_index), event_id, event_time, 96, 733,
    )
    assert high.delay_ticks.float().mean() < low.delay_ticks.float().mean()
    assert high.churned.float().mean() < low.churned.float().mean()
