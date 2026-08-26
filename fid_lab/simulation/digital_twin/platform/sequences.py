"""One chronological point-in-time user sequence authority."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts import EventType
from .projection import PlatformProjectionState


STRONG_INTEREST_EVENTS = (
    EventType.LONG_VIEW,
    EventType.COMPLETE,
    EventType.CLICK,
    EventType.LIKE,
    EventType.COMMENT,
    EventType.SHARE,
    EventType.FOLLOW,
    EventType.DETAIL,
    EventType.FAVORITE,
    EventType.ADD_CART,
    EventType.ORDER,
    EventType.PAYMENT,
)


@dataclass(frozen=True)
class UserSequenceBatch:
    item_id: torch.Tensor
    event_type: torch.Tensor
    surface: torch.Tensor
    duration_ms: torch.Tensor
    event_time: torch.Tensor
    ingest_time: torch.Tensor
    valid: torch.Tensor

    def strong_mask(self) -> torch.Tensor:
        allowed = torch.tensor(
            [int(value) for value in STRONG_INTEREST_EVENTS],
            device=self.item_id.device,
        )
        return self.valid & torch.isin(self.event_type, allowed)


def resolve_user_sequence(
    state: PlatformProjectionState,
    user_id: torch.Tensor,
    request_time: torch.Tensor,
    *,
    max_length: int | None = None,
) -> UserSequenceBatch:
    """Read the ring in chronological order without future-visible events."""
    width = state.user_history_item.shape[1]
    if max_length is not None and not 0 < max_length <= width:
        raise ValueError("sequence max_length must be within the history ring")
    event_number = (
        state.user_history_cursor[user_id, None]
        - width
        + torch.arange(width, device=user_id.device)[None, :]
    )
    retained = event_number >= 0
    slot = torch.remainder(event_number.clamp_min(0), width)

    def gather(name: str) -> torch.Tensor:
        return torch.gather(getattr(state, name)[user_id], 1, slot)

    item = gather("user_history_item")
    event_type = gather("user_history_event_type")
    surface = gather("user_history_surface")
    duration = gather("user_history_duration_ms")
    event_time = gather("user_history_event_time")
    ingest_time = gather("user_history_ingest_time")
    valid = (
        retained
        & (item >= 0)
        & (event_time <= request_time[:, None])
        & (ingest_time <= request_time[:, None])
    )
    if max_length is not None:
        selected = slice(width - max_length, width)
        item = item[:, selected]
        event_type = event_type[:, selected]
        surface = surface[:, selected]
        duration = duration[:, selected]
        event_time = event_time[:, selected]
        ingest_time = ingest_time[:, selected]
        valid = valid[:, selected]
    return UserSequenceBatch(
        item_id=torch.where(valid, item, torch.full_like(item, -1)),
        event_type=torch.where(
            valid, event_type, torch.full_like(event_type, -1),
        ),
        surface=torch.where(valid, surface, torch.full_like(surface, -1)),
        duration_ms=torch.where(valid, duration, torch.zeros_like(duration)),
        event_time=torch.where(
            valid, event_time, torch.full_like(event_time, -1),
        ),
        ingest_time=torch.where(
            valid, ingest_time, torch.full_like(ingest_time, -1),
        ),
        valid=valid,
    )
