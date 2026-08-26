from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin import (
    EventType,
    ObservableProjection,
    Surface,
    build_public_catalog,
    make_app_events,
)
from fid_lab.simulation.digital_twin.platform.sequences import (
    resolve_user_sequence,
)


def test_sequence_authority_orders_ring_and_filters_future_events():
    catalog = build_public_catalog(
        items=32,
        creators=4,
        merchants=2,
        advertisers=2,
        topics=4,
        countries=2,
        regions_per_country=2,
        embedding_dim=8,
        platform_seed=71,
        device="cpu",
    )
    projection = ObservableProjection(
        2, catalog, history_length=3, feed_exposure_history_length=8,
    )
    projection.ingest(make_app_events(
        torch.tensor([
            int(EventType.LONG_VIEW),
            int(EventType.NEGATIVE),
            int(EventType.LIKE),
            int(EventType.SHARE),
        ]),
        event_time=torch.tensor([1, 2, 3, 4]),
        ingest_time=torch.tensor([1, 2, 3, 4]),
        request_id=torch.tensor([10, 20, 30, 40]),
        user_id=torch.zeros(4, dtype=torch.long),
        surface=torch.full((4,), int(Surface.FEED)),
        item_id=torch.tensor([1, 2, 3, 4]),
        creator_id=catalog.creator_id[torch.tensor([1, 2, 3, 4])],
        position=torch.zeros(4, dtype=torch.long),
        ordinal=torch.zeros(4, dtype=torch.long),
    ))
    sequence = resolve_user_sequence(
        projection.state,
        torch.tensor([0]),
        torch.tensor([3]),
    )

    assert sequence.item_id.tolist() == [[2, 3, -1]]
    assert sequence.event_time.tolist() == [[2, 3, -1]]
    assert sequence.valid.tolist() == [[True, True, False]]
    assert sequence.strong_mask().tolist() == [[False, True, False]]
