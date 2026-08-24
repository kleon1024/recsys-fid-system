"""The only payloads allowed to cross platform/environment boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import ItemKind, Surface


TASKS = (
    "play", "play_3s", "long_view", "complete", "like", "comment",
    "share", "follow", "click", "detail", "favorite", "add_cart",
    "order", "payment", "negative", "create", "publish",
)
TASK_INDEX = {name: index for index, name in enumerate(TASKS)}


@dataclass(frozen=True)
class ServedSlate:
    """What the app actually rendered; no unexposed candidates or scores."""

    exposed_item_ids: torch.Tensor


@dataclass
class ObservableResponse:
    """Events emitted by the hidden user world and observable by the app."""

    task: torch.Tensor
    task_mask: torch.Tensor
    stay_seconds: torch.Tensor
    selected_item: torch.Tensor
    active: torch.Tensor

    def event(self, name: str) -> torch.Tensor:
        return self.task[:, TASK_INDEX[name]]


def task_applicability(
    surface: torch.Tensor, kind: torch.Tensor,
) -> torch.Tensor:
    """Single authority for which labels are defined for a served item."""
    mask = torch.zeros(
        len(surface), len(TASKS), device=surface.device, dtype=torch.bool
    )
    feed = surface == int(Surface.FEED)
    search = surface == int(Surface.SEARCH)
    commerce = surface == int(Surface.COMMERCE)
    live = surface == int(Surface.LIVE)
    local = surface == int(Surface.LOCAL)
    posting = surface == int(Surface.POSTING)
    media = kind <= int(ItemKind.LIVE_ROOM)
    for name in ("play", "play_3s", "long_view", "complete"):
        mask[:, TASK_INDEX[name]] = (feed | live | search) & media
    for name in ("like", "comment", "share", "follow"):
        mask[:, TASK_INDEX[name]] = feed | live
    mask[:, TASK_INDEX["click"]] = True
    mask[:, TASK_INDEX["detail"]] = search | commerce | local
    mask[:, TASK_INDEX["favorite"]] = local
    mask[:, TASK_INDEX["add_cart"]] = commerce | live
    mask[:, TASK_INDEX["order"]] = search | commerce | live | local
    mask[:, TASK_INDEX["payment"]] = commerce | live | local
    mask[:, TASK_INDEX["negative"]] = ~posting
    mask[:, TASK_INDEX["create"]] = posting
    mask[:, TASK_INDEX["publish"]] = posting
    return mask
