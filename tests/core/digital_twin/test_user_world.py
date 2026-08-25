from __future__ import annotations

from dataclasses import fields, replace

import torch

from fid_lab.simulation.digital_twin import (
    AtomicSimulationKernel,
    EventType,
    ExperimentPlan,
    ObservableEventLog,
    ObservableProjection,
    RenderedSlateBatch,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
)
from fid_lab.simulation.digital_twin.platform.projection import USER_COUNTER_EVENTS
from fid_lab.simulation.digital_twin.platform.requests import open_platform_requests
from fid_lab.simulation.digital_twin.scenarios.commerce import audit_commerce_funnel
from fid_lab.simulation.digital_twin.world.behavior import sample_response_tensors


def test_tiny_public_catalog_has_safe_missing_business_anchors():
    catalog = build_public_catalog(
        items=1,
        creators=1,
        merchants=1,
        advertisers=1,
        topics=1,
        countries=1,
        regions_per_country=1,
        embedding_dim=2,
        platform_seed=3,
        device="cpu",
    )
    assert catalog.product_id.tolist() == [-1]
    assert catalog.poi_id.tolist() == [-1]


def build_world(users=256, items=720):
    catalog = build_public_catalog(
        items=items,
        creators=120,
        merchants=60,
        topics=24,
        countries=6,
        regions_per_country=8,
        embedding_dim=16,
        platform_seed=71,
        device="cpu",
    )
    config = UserWorldConfig(
        users=users,
        topics=24,
        embedding_dim=16,
        countries=6,
        regions_per_country=8,
        environment_seed=97,
        future_signup_fraction=0.0,
    )
    return UserEcosystemWorld(config, catalog), catalog


def event_keys(events, event_type):
    selected = events.event(event_type)
    return set(zip(
        events.request_id[selected].tolist(),
        events.item_id[selected].tolist(),
        strict=True,
    ))


def publish_source_keys(events):
    selected = events.event(EventType.PUBLISH)
    return set(zip(
        events.request_id[selected].tolist(),
        events.source_candidate_id[selected].tolist(),
        strict=True,
    ))


class CatalogPlatform:
    def __init__(self, catalog, width=12, users=None):
        self.catalog = catalog
        self.width = width
        self.events = 0
        self.projection = (
            None if users is None else ObservableProjection(users, catalog)
        )

    def ingest(self, events):
        self.events += len(events.event_id)
        if self.projection is not None:
            self.projection.ingest(events)

    def snapshot(self):
        return self.events

    def open_requests(self, entry_events):
        return open_platform_requests(entry_events)

    def render(
        self, snapshot, requests, policy, experiment_cell,
        assignment_probability=None,
    ):
        del snapshot
        position = torch.arange(self.width)[None].expand(
            len(requests.request_id), -1,
        )
        item = torch.remainder(
            requests.user_id[:, None] * 13 + position * 29 + int(policy),
            len(self.catalog.item_id),
        )
        return RenderedSlateBatch(
            request_id=requests.request_id,
            user_id=requests.user_id,
            surface=requests.surface,
            event_time=requests.event_time,
            item_ids=item,
            positions=position,
            valid=torch.ones_like(item, dtype=torch.bool),
            ui_variant=torch.full_like(requests.user_id, experiment_cell),
            exposure_probability=torch.ones_like(item, dtype=torch.float),
            selection_policy_kind=torch.zeros_like(requests.user_id),
            exploration_rate=torch.zeros_like(requests.user_id, dtype=torch.float),
            slate_log_probability=torch.zeros_like(
                requests.user_id, dtype=torch.float,
            ),
            assignment_probability=(
                torch.ones(len(requests.request_id))
                if assignment_probability is None
                else assignment_probability
            ),
        )


def test_public_catalog_is_deterministic_and_contains_no_hidden_truth():
    _, left = build_world(users=8, items=90)
    _, right = build_world(users=8, items=90)
    for field in fields(left):
        torch.testing.assert_close(
            getattr(left, field.name), getattr(right, field.name),
        )
    names = {field.name for field in fields(left)}
    assert not {"true_quality", "risk", "latent", "utility"} & names


def test_world_emits_observable_session_and_cascade_events_only():
    world, catalog = build_world()
    entry = world.schedule(0)
    assert int(entry.event(EventType.REGISTRATION).sum()) == 256
    assert int(entry.event(EventType.SURFACE_ENTRY).sum()) == 256
    world.commit(entry)
    requests = CatalogPlatform(catalog).open_requests(entry)
    slate = CatalogPlatform(catalog).render(0, requests, 3, 1)
    events = world.respond(world.snapshot(), slate)
    assert int(events.event(EventType.IMPRESSION).sum()) == 256 * 12
    assert event_keys(events, EventType.EXAMINE) <= event_keys(
        events, EventType.IMPRESSION,
    )
    assert event_keys(events, EventType.PLAY_3S) <= event_keys(
        events, EventType.PLAY,
    )
    assert event_keys(events, EventType.LONG_VIEW) <= event_keys(
        events, EventType.PLAY_3S,
    )
    assert event_keys(events, EventType.COMPLETE) <= event_keys(
        events, EventType.LONG_VIEW,
    )
    assert event_keys(events, EventType.PAYMENT) <= event_keys(
        events, EventType.ORDER,
    )
    assert publish_source_keys(events) <= event_keys(
        events, EventType.CREATE,
    )
    published = events.event(EventType.PUBLISH)
    assert torch.equal(events.item_id[published], events.post_id[published])
    assert (
        events.item_id[published] != events.source_candidate_id[published]
    ).all()


def run_real_world(cell_order):
    world, catalog = build_world()
    platform = CatalogPlatform(catalog)
    log = ObservableEventLog(allowed_lateness=world.max_reporting_lag)
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=0,
        treatment_policy=7,
        experiment_seed=109,
        control_fraction=0.2,
        treatment_fraction=0.2,
    )
    result = AtomicSimulationKernel(world, platform, log).step(
        0, plan, cell_order=cell_order,
    )
    return world, log, result


def test_real_world_ab_is_invariant_to_gpu_cell_execution_order():
    left_world, left_log, left_result = run_real_world((-1, 0, 1))
    right_world, right_log, right_result = run_real_world((1, 0, -1))
    for field in fields(left_world.users):
        torch.testing.assert_close(
            getattr(left_world.users, field.name),
            getattr(right_world.users, field.name),
        )
    left, right = left_log.read(), right_log.read()
    for field in fields(left):
        assert torch.equal(getattr(left, field.name), getattr(right, field.name))
    assert left_result.cell_counts == right_result.cell_counts
    assert left_result.baseline_requests > left_result.experiment_requests


def test_factual_response_changes_the_next_world_snapshot():
    world, catalog = build_world(users=64, items=180)
    platform = CatalogPlatform(catalog)
    kernel = AtomicSimulationKernel(
        world,
        platform,
        ObservableEventLog(allowed_lateness=world.max_reporting_lag),
    )
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=0,
        treatment_policy=11,
        experiment_seed=113,
        control_fraction=0.1,
        treatment_fraction=0.1,
    )
    before = world.users.satisfaction.clone()
    first = kernel.step(0, plan)
    after = world.users.satisfaction.clone()
    assert not torch.equal(before, after)
    second_entry = world.schedule(1)
    assert len(second_entry.event_id) > 0
    assert first.baseline_requests > first.experiment_requests


def test_repeated_feed_video_is_slid_and_degrades_hidden_experience():
    world, catalog = build_world(users=2_048, items=4_000)
    entry = world.schedule(0)
    world.commit(entry)
    requests = CatalogPlatform(catalog).open_requests(entry)
    slate = CatalogPlatform(catalog, width=8).render(0, requests, 3, 0)
    feed = slate.surface == 0
    slate = slate.select(feed)
    video = torch.where(catalog.content_kind == 0)[0]
    slate = replace(
        slate,
        item_ids=video[torch.remainder(slate.item_ids, len(video))],
    )
    fresh = sample_response_tensors(
        world.snapshot(), catalog, slate, world.config.environment_seed,
    )
    first_events = world.respond(world.snapshot(), slate)
    world.commit(first_events)
    satisfaction_before_repeat = world.users.satisfaction[slate.user_id].clone()
    fatigue_before_repeat = world.users.fatigue[slate.user_id].clone()
    repeated = sample_response_tensors(
        world.snapshot(), catalog, slate, world.config.environment_seed,
    )
    assert float(repeated.utility.mean()) < float(fresh.utility.mean()) - 2.0
    assert float(repeated.action[EventType.PLAY].float().mean()) < (
        0.1 * float(fresh.action[EventType.PLAY].float().mean())
    )
    assert float(repeated.action[EventType.SLIDE].float().mean()) > (
        float(fresh.action[EventType.SLIDE].float().mean())
    )
    repeated_events = world.respond(world.snapshot(), slate)
    assert int(repeated_events.event(EventType.SESSION_END).sum()) > int(
        first_events.event(EventType.SESSION_END).sum()
    )
    world.commit(repeated_events)
    assert float(world.users.satisfaction[slate.user_id].mean()) < float(
        satisfaction_before_repeat.mean()
    )
    assert float(world.users.fatigue[slate.user_id].mean()) > float(
        fatigue_before_repeat.mean()
    )


def test_kernel_delivers_delayed_funnel_into_point_in_time_projection():
    world, catalog = build_world(users=512, items=1_800)
    platform = CatalogPlatform(catalog, users=512)
    log = ObservableEventLog(allowed_lateness=world.max_reporting_lag)
    kernel = AtomicSimulationKernel(world, platform, log)
    plan = ExperimentPlan.ramped_user_ab(
        active_policy=0,
        treatment_policy=7,
        experiment_seed=109,
        control_fraction=0.05,
        treatment_fraction=0.05,
    )
    for logical_time in range(30):
        kernel.step(logical_time, plan)
    events = log.read()
    orders = events.event(EventType.ORDER)
    payments = events.event(EventType.PAYMENT)
    assert int(orders.sum()) > 0
    assert int(payments.sum()) > 0
    assert set(events.order_id[payments].tolist()) <= set(
        events.order_id[orders].tolist()
    )
    assert (events.event_time[payments] > 0).all()
    order_column = USER_COUNTER_EVENTS.index(EventType.ORDER)
    projected_orders = platform.projection.state.user_event_counts[
        :, order_column
    ].sum()
    assert int(projected_orders) == int(orders.sum())
    commerce = audit_commerce_funnel(events)
    assert commerce.impressions >= commerce.clicks >= commerce.details
    assert commerce.details >= commerce.carts >= commerce.orders
    assert commerce.carts > 0
    assert commerce.orders >= commerce.payments
