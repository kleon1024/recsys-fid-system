from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin import (
    ContentKind,
    EventType,
    build_public_catalog,
    make_app_events,
)
from fid_lab.simulation.digital_twin.contracts import AppEventBatch
from fid_lab.simulation.digital_twin.world.supply import SupplyEcosystem


def build_supply():
    catalog = build_public_catalog(
        items=2_400,
        creators=160,
        merchants=80,
        topics=32,
        countries=8,
        regions_per_country=6,
        embedding_dim=16,
        platform_seed=211,
        device="cpu",
        advertisers=40,
        initial_active_fraction=0.8,
    )
    return SupplyEcosystem(catalog, environment_seed=223, ticks_per_day=96)


def item_event(supply, event_type, item, request_id):
    catalog = supply.catalog
    item_id = torch.tensor([item])
    return make_app_events(
        event_type,
        event_time=5,
        request_id=torch.tensor([request_id]),
        user_id=torch.tensor([3]),
        surface=torch.tensor([0]),
        item_id=item_id,
        creator_id=catalog.creator_id[item_id],
        merchant_id=catalog.merchant_id[item_id],
        advertiser_id=catalog.advertiser_id[item_id],
        content_kind=catalog.content_kind[item_id],
        position=torch.tensor([0]),
        experiment_cell=torch.tensor([1]),
        logging_probability=torch.tensor([1.0]),
        assignment_probability=torch.tensor([0.05]),
    )


def test_creator_agents_publish_reserved_items_as_observable_events():
    supply = build_supply()
    supply.state.creator_next_publish.zero_()
    supply.state.creator_motivation.fill_(1.0)
    supply.state.creator_cost.zero_()
    before = int(supply.state.item_active.sum())
    events = supply.schedule(0)
    published = events.event(EventType.PUBLISH)
    assert int(published.sum()) > 0
    assert (events.user_id[published] == -1).all()
    assert not supply.state.item_active[events.item_id[published]].any()
    supply.commit(events)
    assert int(supply.state.item_active.sum()) == before + int(published.sum())
    assert supply.state.item_active[events.item_id[published]].all()


def test_orders_ads_and_creator_feedback_change_only_hidden_supply_state():
    supply = build_supply()
    catalog, state = supply.catalog, supply.state
    product = int(torch.where(
        state.item_active
        & (catalog.content_kind == int(ContentKind.PRODUCT))
    )[0][0])
    ad = int(torch.where(
        state.item_active
        & (catalog.content_kind == int(ContentKind.AD))
    )[0][0])
    creator = int(catalog.creator_id[product])
    advertiser = int(catalog.advertiser_id[ad])
    state.creator_motivation[creator] = 0.5
    inventory_before = float(state.item_inventory[product])
    budget_before = float(state.advertiser_budget[advertiser])
    events = AppEventBatch.concatenate((
        item_event(supply, EventType.IMPRESSION, product, 501),
        item_event(supply, EventType.LIKE, product, 501),
        item_event(supply, EventType.ORDER, product, 501),
        item_event(supply, EventType.IMPRESSION, ad, 503),
    ))
    supply.commit(events)
    assert float(state.creator_motivation[creator]) > 0.5
    assert float(state.item_inventory[product]) < inventory_before
    assert float(state.advertiser_budget[advertiser]) < budget_before
