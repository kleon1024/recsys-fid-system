from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin import (
    ContentKind,
    EventType,
    build_public_catalog,
    make_app_events,
)
from fid_lab.simulation.digital_twin.contracts import (
    AppEventBatch,
    PublishFailureReason,
)
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


def test_posting_intent_materializes_one_immutable_reserved_post():
    supply = build_supply()
    catalog = supply.catalog
    post_kind = (
        (catalog.content_kind == int(ContentKind.SHORT_VIDEO))
        | (catalog.content_kind == int(ContentKind.PHOTO))
        | (catalog.content_kind == int(ContentKind.ARTICLE))
        | (catalog.content_kind == int(ContentKind.CARD))
    )
    reserved = torch.where(~supply.state.item_active & post_kind)[0][0]
    creator = supply.state.item_creator_id[reserved:reserved + 1]
    source = torch.where(
        supply.state.item_active
        & (catalog.content_kind == int(ContentKind.POI))
    )[0][:1]
    intent = make_app_events(
        EventType.PUBLISH,
        event_time=0,
        request_id=torch.tensor([701]),
        user_id=torch.tensor([3]),
        surface=torch.tensor([5]),
        item_id=source,
        source_candidate_id=source,
        position=torch.tensor([0]),
        creator_id=creator,
        poi_id=source,
        content_kind=catalog.content_kind[source],
        topic_id=catalog.topic_id[source],
        country=torch.tensor([0]),
        region=torch.tensor([1]),
        assignment_probability=torch.tensor([0.1]),
        logging_probability=torch.tensor([1.0]),
    )
    before = int(supply.state.item_active.sum())
    events = supply.materialize_user_posts(intent)
    published = events.event(EventType.PUBLISH)
    assert int(published.sum()) == 1
    assert torch.equal(events.post_id[published], events.item_id[published])
    assert torch.equal(events.source_candidate_id[published], source)
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


def test_publish_failure_distinguishes_capacity_and_creator_exit():
    supply = build_supply()
    catalog = supply.catalog
    post_kind = catalog.content_kind <= int(ContentKind.CARD)
    creator = int(supply.state.item_creator_id[torch.where(post_kind)[0][0]])
    supply.state.item_active[
        post_kind & (supply.state.item_creator_id == creator)
    ] = True
    source = torch.where(
        supply.state.item_active
        & (catalog.content_kind == int(ContentKind.POI))
    )[0][:1]

    def intent(request_id: int):
        return make_app_events(
            EventType.PUBLISH,
            event_time=0,
            request_id=torch.tensor([request_id]),
            user_id=torch.tensor([3]),
            surface=torch.tensor([5]),
            item_id=source,
            source_candidate_id=source,
            position=torch.tensor([0]),
            creator_id=torch.tensor([creator]),
            content_kind=catalog.content_kind[source],
            topic_id=catalog.topic_id[source],
        )

    no_capacity = supply.materialize_user_posts(intent(801))
    assert no_capacity.event(EventType.PUBLISH_FAILED).all()
    assert int(no_capacity.value[0]) == int(PublishFailureReason.NO_CAPACITY)
    supply.state.creator_retained[creator] = False
    exited = supply.materialize_user_posts(intent(803))
    assert exited.event(EventType.PUBLISH_FAILED).all()
    assert int(exited.value[0]) == int(PublishFailureReason.CREATOR_EXITED)


def test_moderation_deletion_and_creator_exit_remove_future_supply():
    supply = build_supply()
    state, catalog = supply.state, supply.catalog
    post = torch.where(
        state.item_active & (catalog.content_kind <= int(ContentKind.CARD))
    )[0]
    moderated, deleted = int(post[0]), int(post[1])
    exit_creator = int(state.item_creator_id[deleted])
    state.item_moderation_risk.zero_()
    state.item_moderation_risk[moderated] = 1.0
    state.item_delete_propensity.zero_()
    state.item_delete_propensity[deleted] = 1.0
    state.creator_motivation[exit_creator] = 0.0
    state.item_publish_time[deleted] = 0
    state.creator_last_publish[exit_creator] = 0
    state.creator_last_feedback[exit_creator] = 0
    logical_time = 30 * supply.ticks_per_day
    events = supply.schedule(logical_time)
    assert (
        events.item_id[events.event(EventType.MODERATION_REMOVE)] == moderated
    ).any()
    assert (
        events.item_id[events.event(EventType.CONTENT_DELETE)] == deleted
    ).any()
    assert (
        events.creator_id[events.event(EventType.CREATOR_EXIT)] == exit_creator
    ).any()
    supply.commit(events)
    assert not bool(state.item_active[moderated])
    assert not bool(state.item_active[deleted])
    assert not bool(state.creator_retained[exit_creator])


def test_factual_feedback_precedes_same_creator_future_supply():
    supply = build_supply()
    catalog, state = supply.catalog, supply.state
    post_kind = catalog.content_kind <= int(ContentKind.CARD)
    reserved = ~state.item_active & post_kind
    counts = torch.bincount(
        state.item_creator_id[reserved],
        minlength=len(state.creator_motivation),
    )
    creator = int(torch.where(counts >= 2)[0][0])
    source = torch.where(
        state.item_active & (catalog.content_kind == int(ContentKind.POI))
    )[0][:1]

    def intent(request_id: int, event_time: int):
        return make_app_events(
            EventType.PUBLISH,
            event_time=event_time,
            request_id=torch.tensor([request_id]),
            user_id=torch.tensor([3]),
            surface=torch.tensor([5]),
            item_id=source,
            source_candidate_id=source,
            position=torch.tensor([0]),
            creator_id=torch.tensor([creator]),
            content_kind=catalog.content_kind[source],
            topic_id=catalog.topic_id[source],
        )

    first = supply.materialize_user_posts(intent(901, 0))
    supply.commit(first)
    first_post = int(first.post_id[first.event(EventType.PUBLISH)][0])
    motivation_after_publish = float(state.creator_motivation[creator])
    feedback = AppEventBatch.concatenate((
        item_event(supply, EventType.IMPRESSION, first_post, 903),
        item_event(supply, EventType.LIKE, first_post, 903),
    ))
    supply.commit(feedback)
    assert float(state.creator_motivation[creator]) > motivation_after_publish
    state.creator_next_publish[creator] = 6
    second = supply.materialize_user_posts(intent(905, 6))
    supply.commit(second)
    second_post = int(second.post_id[second.event(EventType.PUBLISH)][0])
    assert second_post != first_post
    assert int(state.creator_last_publish[creator]) == 6
