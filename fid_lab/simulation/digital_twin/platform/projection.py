"""Point-in-time platform state derived only from delivered app events."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..catalog import PublicCatalog
from ..contracts import AppEventBatch, EventType, Surface


USER_COUNTER_EVENTS = (
    EventType.IMPRESSION,
    EventType.EXAMINE,
    EventType.PLAY,
    EventType.PLAY_3S,
    EventType.LONG_VIEW,
    EventType.COMPLETE,
    EventType.CLICK,
    EventType.LIKE,
    EventType.SHARE,
    EventType.NEGATIVE,
    EventType.ORDER,
    EventType.PAYMENT,
    EventType.REFUND,
    EventType.PIXEL_CONVERSION,
    EventType.PUBLISH,
)

ITEM_COUNTER_EVENTS = (
    EventType.IMPRESSION,
    EventType.EXAMINE,
    EventType.LONG_VIEW,
    EventType.CLICK,
    EventType.NEGATIVE,
    EventType.ORDER,
    EventType.PAYMENT,
    EventType.REFUND,
)


@dataclass
class PlatformProjectionState:
    user_registered: torch.Tensor
    user_country: torch.Tensor
    user_region: torch.Tensor
    user_last_event_time: torch.Tensor
    user_last_ingest_time: torch.Tensor
    user_event_counts: torch.Tensor
    user_surface_counts: torch.Tensor
    user_history_item: torch.Tensor
    user_history_event_time: torch.Tensor
    user_history_ingest_time: torch.Tensor
    user_history_cursor: torch.Tensor
    item_active: torch.Tensor
    item_publish_time: torch.Tensor
    item_inventory: torch.Tensor
    item_bid: torch.Tensor
    item_event_counts: torch.Tensor
    creator_impressions: torch.Tensor
    creator_engagements: torch.Tensor
    last_ingest_time: torch.Tensor

    def clone(self) -> PlatformProjectionState:
        return PlatformProjectionState(**{
            field.name: getattr(self, field.name).clone()
            for field in fields(self)
        })


@dataclass(frozen=True)
class ProjectionSnapshot:
    state: PlatformProjectionState
    as_of_ingest_time: int


def build_projection_state(
    users: int,
    catalog: PublicCatalog,
    history_length: int,
) -> PlatformProjectionState:
    if users <= 0 or history_length <= 0:
        raise ValueError("projection dimensions must be positive")
    device = catalog.item_id.device
    integer_missing = torch.full((users,), -1, device=device, dtype=torch.long)
    creators = int(catalog.creator_id.max()) + 1
    return PlatformProjectionState(
        user_registered=torch.zeros(users, device=device, dtype=torch.bool),
        user_country=integer_missing.clone(),
        user_region=integer_missing.clone(),
        user_last_event_time=integer_missing.clone(),
        user_last_ingest_time=integer_missing.clone(),
        user_event_counts=torch.zeros(
            users, len(USER_COUNTER_EVENTS), device=device,
        ),
        user_surface_counts=torch.zeros(users, len(Surface), device=device),
        user_history_item=torch.full(
            (users, history_length), -1, device=device, dtype=torch.long,
        ),
        user_history_event_time=torch.full(
            (users, history_length), -1, device=device, dtype=torch.long,
        ),
        user_history_ingest_time=torch.full(
            (users, history_length), -1, device=device, dtype=torch.long,
        ),
        user_history_cursor=torch.zeros(users, device=device, dtype=torch.long),
        item_active=catalog.active.clone(),
        item_publish_time=catalog.publish_time.clone(),
        item_inventory=catalog.inventory.clone(),
        item_bid=torch.zeros(len(catalog.item_id), device=device),
        item_event_counts=torch.zeros(
            len(catalog.item_id), len(ITEM_COUNTER_EVENTS), device=device,
        ),
        creator_impressions=torch.zeros(creators, device=device),
        creator_engagements=torch.zeros(creators, device=device),
        last_ingest_time=torch.tensor(-1, device=device, dtype=torch.long),
    )


class ObservableProjection:
    """Online feature authority; hidden world state is structurally unavailable."""

    def __init__(
        self, users: int, catalog: PublicCatalog, history_length: int = 128,
    ):
        self.catalog = catalog
        self.state = build_projection_state(users, catalog, history_length)

    def ingest(self, events: AppEventBatch) -> None:
        if not len(events.event_id):
            return
        if int(events.ingest_time.min()) < int(self.state.last_ingest_time):
            raise ValueError("projection cannot ingest delivery time out of order")
        self._profiles(events)
        self._user_counters(events)
        self._item_counters(events)
        self._supply(events)
        self._history(events)
        self.state.last_ingest_time.fill_(int(events.ingest_time.max()))

    def snapshot(self) -> ProjectionSnapshot:
        return ProjectionSnapshot(
            self.state.clone(), int(self.state.last_ingest_time),
        )

    def view(self) -> ProjectionSnapshot:
        """Immutable-by-kernel-lifetime view used during one serving microbatch."""
        return ProjectionSnapshot(
            self.state, int(self.state.last_ingest_time),
        )

    def _profiles(self, events: AppEventBatch) -> None:
        valid = events.user_id >= 0
        user = events.user_id[valid]
        registration = events.event_type[valid] == int(EventType.REGISTRATION)
        self.state.user_registered[user[registration]] = True
        profile_event = (
            events.event(EventType.REGISTRATION)
            | events.event(EventType.SESSION_START)
            | events.event(EventType.SURFACE_ENTRY)
            | events.event(EventType.QUERY)
        )[valid]
        country_known = profile_event & (events.country[valid] >= 0)
        self.state.user_country[user[country_known]] = events.country[valid][
            country_known
        ]
        region_known = profile_event & (events.region[valid] >= 0)
        self.state.user_region[user[region_known]] = events.region[valid][
            region_known
        ]
        self._scatter_max(
            self.state.user_last_event_time,
            user,
            events.event_time[valid],
        )
        self._scatter_max(
            self.state.user_last_ingest_time,
            user,
            events.ingest_time[valid],
        )

    def _user_counters(self, events: AppEventBatch) -> None:
        valid_user = events.user_id >= 0
        for column, event_type in enumerate(USER_COUNTER_EVENTS):
            selected = valid_user & events.event(event_type)
            self.state.user_event_counts[:, column].index_add_(
                0,
                events.user_id[selected],
                torch.ones_like(events.user_id[selected], dtype=torch.float),
            )
        for surface in Surface:
            selected = (
                valid_user
                & events.event(EventType.SURFACE_ENTRY)
                & (events.surface == int(surface))
            )
            self.state.user_surface_counts[:, int(surface)].index_add_(
                0,
                events.user_id[selected],
                torch.ones_like(events.user_id[selected], dtype=torch.float),
            )

    def _item_counters(self, events: AppEventBatch) -> None:
        valid_item = events.item_id >= 0
        for column, event_type in enumerate(ITEM_COUNTER_EVENTS):
            selected = valid_item & events.event(event_type)
            self.state.item_event_counts[:, column].index_add_(
                0,
                events.item_id[selected],
                torch.ones_like(events.item_id[selected], dtype=torch.float),
            )
        impression = valid_item & events.event(EventType.IMPRESSION)
        self.state.creator_impressions.index_add_(
            0,
            events.creator_id[impression],
            torch.ones_like(events.creator_id[impression], dtype=torch.float),
        )
        engagement = valid_item & (
            events.event(EventType.LONG_VIEW)
            | events.event(EventType.LIKE)
            | events.event(EventType.SHARE)
            | events.event(EventType.FOLLOW)
            | events.event(EventType.ORDER)
            | events.event(EventType.PAYMENT)
        )
        self.state.creator_engagements.index_add_(
            0,
            events.creator_id[engagement],
            torch.ones_like(events.creator_id[engagement], dtype=torch.float),
        )

    def _supply(self, events: AppEventBatch) -> None:
        publish = (
            events.event(EventType.PUBLISH)
            & (events.user_id < 0)
            & (events.item_id >= 0)
        )
        item = events.item_id[publish]
        self.state.item_active[item] = True
        self.state.item_publish_time[item] = events.event_time[publish]
        inventory = events.event(EventType.INVENTORY) & (events.item_id >= 0)
        self.state.item_inventory[events.item_id[inventory]] = events.value[
            inventory
        ]
        bid = events.event(EventType.BID) & (events.item_id >= 0)
        self.state.item_bid[events.item_id[bid]] = events.value[bid]

    def _history(self, events: AppEventBatch) -> None:
        selected = (
            events.event(EventType.DWELL)
            & (events.user_id >= 0)
            & (events.item_id >= 0)
        )
        if not selected.any():
            return
        user = events.user_id[selected]
        item = events.item_id[selected]
        event_time = events.event_time[selected]
        ingest_time = events.ingest_time[selected]
        order = torch.argsort(user, stable=True)
        user, item = user[order], item[order]
        event_time, ingest_time = event_time[order], ingest_time[order]
        new_user = torch.ones_like(user, dtype=torch.bool)
        new_user[1:] = user[1:] != user[:-1]
        starts = torch.where(new_user)[0]
        group = torch.cumsum(new_user.long(), dim=0) - 1
        within = torch.arange(len(user), device=user.device) - starts[group]
        ends = torch.cat((starts[1:], torch.tensor(
            [len(user)], device=user.device,
        )))
        group_size = (ends - starts)[group]
        history_length = self.state.user_history_item.shape[1]
        keep = within >= (group_size - history_length).clamp_min(0)
        slot = torch.remainder(
            self.state.user_history_cursor[user] + within,
            history_length,
        )
        self.state.user_history_item[user[keep], slot[keep]] = item[keep]
        self.state.user_history_event_time[user[keep], slot[keep]] = event_time[
            keep
        ]
        self.state.user_history_ingest_time[user[keep], slot[keep]] = ingest_time[
            keep
        ]
        counts = torch.zeros_like(self.state.user_history_cursor)
        counts.index_add_(
            0, user, torch.ones_like(user, dtype=torch.long),
        )
        self.state.user_history_cursor.add_(counts)

    @staticmethod
    def _scatter_max(
        target: torch.Tensor, index: torch.Tensor, source: torch.Tensor,
    ) -> None:
        target.scatter_reduce_(
            0, index, source, reduce="amax", include_self=True,
        )
