from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin.contracts import (
    EventType,
    Surface,
    make_app_events,
)
from fid_lab.simulation.digital_twin.platform.indexes.graph import (
    CoVisitGraphIndex,
)


def test_graph_uses_qualified_feed_dwell_instead_of_co_exposure():
    events = make_app_events(
        EventType.DWELL,
        event_time=1,
        request_id=torch.tensor([10, 10, 20, 20, 30, 30]),
        user_id=torch.tensor([0, 0, 1, 1, 2, 2]),
        surface=torch.tensor([
            int(Surface.FEED), int(Surface.FEED),
            int(Surface.FEED), int(Surface.FEED),
            int(Surface.SEARCH), int(Surface.SEARCH),
        ]),
        item_id=torch.tensor([1, 2, 3, 4, 5, 6]),
        position=torch.tensor([0, 1, 0, 1, 0, 1]),
        ordinal=torch.tensor([0, 1, 0, 1, 0, 1]),
        duration_ms=torch.tensor([1_000, 5_000, 4_000, 5_000, 6_000, 7_000]),
    )
    graph = CoVisitGraphIndex(16, 4, torch.device("cpu"))
    graph.update(events)
    graph.refresh("qualified-v1")

    assert graph.neighbor[1].eq(-1).all()
    assert graph.neighbor[3, 0].item() == 4
    assert graph.neighbor[4, 0].item() == 3
    assert graph.neighbor[5].eq(-1).all()
