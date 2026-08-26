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


def events(event_type, request, user, item, surface=Surface.FEED):
    rows = len(request)
    return make_app_events(
        event_type,
        event_time=1,
        request_id=torch.tensor(request),
        user_id=torch.tensor(user),
        surface=torch.full((rows,), int(surface)),
        item_id=torch.tensor(item),
        position=torch.zeros(rows, dtype=torch.long),
        ordinal=torch.arange(rows),
    )


def test_graph_uses_cross_request_strong_actions_and_ignores_weak_events():
    graph = CoVisitGraphIndex(16, 4, torch.device("cpu"))
    graph.update(events(EventType.LONG_VIEW, [10, 11], [0, 0], [1, 2]))
    graph.update(events(EventType.LIKE, [12], [0], [3]))
    graph.update(events(EventType.NEGATIVE, [20, 21], [1, 1], [4, 5]))
    graph.update(events(
        EventType.FAVORITE, [30, 31], [2, 2], [6, 7], Surface.SEARCH,
    ))
    graph.refresh("qualified-v1")

    assert graph.neighbor[1, 0].item() == 2
    assert set(graph.neighbor[2, :2].tolist()) == {1, 3}
    assert graph.neighbor[3, 0].item() == 2
    assert graph.neighbor[4].eq(-1).all()
    assert graph.neighbor[6].eq(-1).all()
