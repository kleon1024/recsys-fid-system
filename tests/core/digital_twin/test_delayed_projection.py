from __future__ import annotations

import ast
from pathlib import Path

import torch

from fid_lab.simulation.digital_twin import (
    AppEventBatch,
    ContentKind,
    EventType,
    ObservableEventLog,
    ObservableProjection,
    UserEcosystemWorld,
    UserWorldConfig,
    build_public_catalog,
    make_app_events,
)
from fid_lab.simulation.digital_twin.platform.projection import (
    USER_COUNTER_EVENTS,
)


def build_world():
    catalog = build_public_catalog(
        items=3_600,
        creators=180,
        merchants=90,
        advertisers=45,
        topics=24,
        countries=6,
        regions_per_country=8,
        embedding_dim=16,
        platform_seed=307,
        device="cpu",
    )
    world = UserEcosystemWorld(UserWorldConfig(
        users=512,
        topics=24,
        embedding_dim=16,
        countries=6,
        regions_per_country=8,
        environment_seed=311,
        future_signup_fraction=0.0,
    ), catalog)
    return world, catalog


def precursor_events(world, event_type, item, rows, surface):
    catalog = world.catalog
    request = torch.arange(rows) + 10_000
    user = torch.remainder(torch.arange(rows), len(world.users.user_id))
    item_id = torch.full((rows,), item, dtype=torch.long)
    return make_app_events(
        event_type,
        event_time=0,
        request_id=request,
        user_id=user,
        surface=torch.full((rows,), surface, dtype=torch.long),
        item_id=item_id,
        creator_id=catalog.creator_id[item_id],
        merchant_id=catalog.merchant_id[item_id],
        advertiser_id=catalog.advertiser_id[item_id],
        product_id=torch.where(
            catalog.content_kind[item_id] == int(ContentKind.PRODUCT),
            item_id,
            torch.full_like(item_id, -1),
        ),
        poi_id=torch.where(
            catalog.content_kind[item_id] == int(ContentKind.POI),
            item_id,
            torch.full_like(item_id, -1),
        ),
        content_kind=catalog.content_kind[item_id],
        topic_id=catalog.topic_id[item_id],
        country=world.users.country[user],
        region=world.users.region[user],
        position=torch.zeros(rows, dtype=torch.long),
        experiment_cell=torch.ones(rows, dtype=torch.long),
        logging_probability=torch.ones(rows),
        assignment_probability=torch.full((rows,), 0.05),
    )


def test_delayed_commerce_chain_is_scheduled_and_acknowledged_by_ingest_time():
    world, catalog = build_world()
    product = int(torch.where(
        catalog.content_kind == int(ContentKind.PRODUCT)
    )[0][0])
    details = precursor_events(world, EventType.DETAIL, product, 4_000, 2)
    queue = world.delayed
    queue.schedule_from(
        details, world.users, world.catalog_truth, world.supply.state,
    )
    assert queue.pending_events > 0
    order_batches = tuple(queue.due(step) for step in range(1, 13))
    orders = AppEventBatch.concatenate(order_batches)
    assert len(orders.event_id) > 0
    assert orders.event(EventType.ORDER).all()
    assert (orders.event_time == orders.ingest_time).all()
    first_due = int(orders.ingest_time.min())
    first_delivery = queue.due(first_due)
    assert torch.equal(first_delivery.event_id, queue.due(first_due).event_id)
    pending_before = queue.pending_events
    queue.acknowledge(first_delivery)
    assert queue.pending_events < pending_before
    queue.schedule_from(
        first_delivery,
        world.users,
        world.catalog_truth,
        world.supply.state,
    )
    later = AppEventBatch.concatenate(tuple(
        queue.due(step) for step in range(first_due + 1, first_due + 6)
    ))
    assert later.event(EventType.PAYMENT).any()


def test_pixel_event_preserves_occurrence_time_and_late_delivery_time():
    world, catalog = build_world()
    ad = int(torch.where(
        catalog.content_kind == int(ContentKind.AD)
    )[0][0])
    clicks = precursor_events(world, EventType.CLICK, ad, 8_000, 0)
    world.delayed.schedule_from(
        clicks, world.users, world.catalog_truth, world.supply.state,
    )
    delivered = AppEventBatch.concatenate(tuple(
        world.delayed.due(step)
        for step in range(1, 3 * world.config.ticks_per_day + 1)
    ))
    pixel = delivered.event(EventType.PIXEL_CONVERSION)
    assert pixel.any()
    assert (delivered.ingest_time[pixel] >= delivered.event_time[pixel]).all()
    assert (
        delivered.ingest_time[pixel] > delivered.event_time[pixel]
    ).any()


def simple_event(
    event_type, *, event_time, ingest_time, request, user, item=-1,
):
    rows = len(request)
    return make_app_events(
        event_type,
        event_time=event_time,
        ingest_time=ingest_time,
        request_id=request,
        user_id=user,
        surface=torch.zeros(rows, dtype=torch.long),
        item_id=torch.full((rows,), item, dtype=torch.long),
        position=torch.zeros(rows, dtype=torch.long),
        country=torch.full((rows,), 2, dtype=torch.long),
        region=torch.full((rows,), 17, dtype=torch.long),
    )


def test_projection_exposes_late_outcome_only_after_ingestion():
    _, catalog = build_world()
    projection = ObservableProjection(512, catalog, history_length=4)
    registration = simple_event(
        EventType.REGISTRATION,
        event_time=0,
        ingest_time=0,
        request=torch.tensor([1]),
        user=torch.tensor([3]),
    )
    projection.ingest(registration)
    before = projection.snapshot()
    pixel = simple_event(
        EventType.PIXEL_CONVERSION,
        event_time=2,
        ingest_time=5,
        request=torch.tensor([2]),
        user=torch.tensor([3]),
        item=7,
    )
    pixel_column = USER_COUNTER_EVENTS.index(EventType.PIXEL_CONVERSION)
    assert float(before.state.user_event_counts[3, pixel_column]) == 0.0
    projection.ingest(pixel)
    after = projection.snapshot()
    assert float(after.state.user_event_counts[3, pixel_column]) == 1.0
    assert int(after.state.user_last_event_time[3]) == 2
    assert int(after.state.user_last_ingest_time[3]) == 5
    assert int(after.state.user_country[3]) == 2


def test_event_log_separates_ingest_watermark_from_event_time():
    log = ObservableEventLog(allowed_lateness=3)
    immediate = simple_event(
        EventType.CLICK,
        event_time=0,
        ingest_time=0,
        request=torch.tensor([41]),
        user=torch.tensor([3]),
        item=7,
    )
    late = simple_event(
        EventType.PIXEL_CONVERSION,
        event_time=2,
        ingest_time=5,
        request=torch.tensor([43]),
        user=torch.tensor([3]),
        item=7,
    )
    log.append(immediate)
    log.append(late)
    assert log.ingest_watermark == 5
    assert log.watermark == 2
    assert len(log.read(ingested_through=0).event_id) == 1
    assert len(log.read(through=2, ingested_through=5).event_id) == 2


def test_projection_ring_keeps_latest_point_in_time_dwell_history():
    _, catalog = build_world()
    projection = ObservableProjection(512, catalog, history_length=4)
    batches = []
    for index in range(10):
        batches.append(simple_event(
            EventType.DWELL,
            event_time=index,
            ingest_time=10,
            request=torch.tensor([100 + index]),
            user=torch.tensor([5]),
            item=index,
        ))
    projection.ingest(AppEventBatch.concatenate(batches))
    history = projection.state.user_history_item[5]
    assert set(history.tolist()) == {6, 7, 8, 9}
    assert int(projection.state.user_history_cursor[5]) == 10


def test_platform_projection_has_no_import_path_to_hidden_world():
    source = Path(
        "fid_lab/simulation/digital_twin/platform/projection.py"
    ).read_text()
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(module and "world" in module for module in imports)
