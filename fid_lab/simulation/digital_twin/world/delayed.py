"""Event-time queue for delayed commerce, refund and Pixel outcomes."""

from __future__ import annotations

import torch

from ...randomness.counter import uniform
from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    ContentKind,
    EventType,
    Surface,
    make_app_events,
)
from .state import HiddenCatalogTruth, HiddenUserState
from .supply import HiddenSupplyState


class DelayedOutcomeQueue:
    """Pending outcomes are visible only when their ingest time arrives."""

    def __init__(
        self,
        catalog: PublicCatalog,
        environment_seed: int,
        ticks_per_day: int,
    ):
        self.catalog = catalog
        self.seed = environment_seed
        self.ticks_per_day = ticks_per_day
        self._pending: dict[int, AppEventBatch] = {}

    @property
    def pending_events(self) -> int:
        return sum(len(batch.event_id) for batch in self._pending.values())

    def due(self, ingest_time: int) -> AppEventBatch:
        batch = self._pending.get(ingest_time)
        if batch is None:
            return AppEventBatch.empty(self.catalog.item_id.device)
        return batch

    def acknowledge(self, delivered: AppEventBatch) -> None:
        for ingest_time in torch.unique(delivered.ingest_time).tolist():
            pending = self._pending.get(ingest_time)
            if pending is None:
                continue
            delivered_ids = delivered.event_id[
                delivered.ingest_time == ingest_time
            ]
            remaining = ~torch.isin(pending.event_id, delivered_ids)
            if remaining.any():
                self._pending[ingest_time] = pending.select(remaining)
            else:
                del self._pending[ingest_time]

    def schedule_from(
        self,
        events: AppEventBatch,
        users: HiddenUserState,
        truth: HiddenCatalogTruth,
        supply: HiddenSupplyState,
    ) -> None:
        generated = (
            self._orders(events, users, truth, supply),
            self._payments(events, users),
            self._refunds(events, truth, supply),
            self._pixel_conversions(events, users, truth),
        )
        self._add(AppEventBatch.concatenate(generated))

    def _add(self, events: AppEventBatch) -> None:
        for ingest_time in torch.unique(events.ingest_time).tolist():
            selected = events.ingest_time == ingest_time
            incoming = events.select(selected)
            existing = self._pending.get(ingest_time)
            self._pending[ingest_time] = (
                incoming
                if existing is None
                else AppEventBatch.concatenate((existing, incoming))
            )

    def _orders(
        self,
        events: AppEventBatch,
        users: HiddenUserState,
        truth: HiddenCatalogTruth,
        supply: HiddenSupplyState,
    ) -> AppEventBatch:
        commerce_cart = events.event(EventType.ADD_CART) & (
            events.surface == int(Surface.COMMERCE)
        )
        local_submit = events.event(EventType.DETAIL) & (
            events.surface == int(Surface.LOCAL)
        )
        eligible = (
            (commerce_cart | local_submit)
            & (events.item_id >= 0)
            & supply.item_active[events.item_id.clamp_min(0)]
            & (supply.item_inventory[events.item_id.clamp_min(0)] > 0.0)
        )
        if not eligible.any():
            return AppEventBatch.empty(events.event_id.device)
        row = torch.where(eligible)[0]
        user = events.user_id[row]
        item = events.item_id[row]
        affordability = torch.exp(-torch.abs(
            torch.log1p(self.catalog.price[item])
            - 4.0 * users.spending_power[user]
        ))
        probability = torch.sigmoid(
            -3.1
            + 1.25 * truth.price_appeal[item]
            + 0.85 * affordability
            + 0.55 * users.satisfaction[user]
            - 0.8 * users.fatigue[user]
            + 0.35 * supply.item_inventory[item]
        )
        chosen = uniform(
            events.event_id[row], 0, 1_501, self.seed,
        ) < probability
        row = self._within_available_inventory(events, row[chosen], supply)
        delay = 1 + torch.floor(
            12.0 * uniform(
                events.event_id[row], 0, 1_503, self.seed,
            ).square()
        ).long()
        occurrence = events.event_time[row] + delay
        return self._derived(
            EventType.ORDER,
            events,
            row,
            event_time=occurrence,
            ingest_time=occurrence,
            value=self.catalog.price[events.item_id[row]],
        )

    def _within_available_inventory(
        self,
        events: AppEventBatch,
        row: torch.Tensor,
        supply: HiddenSupplyState,
    ) -> torch.Tensor:
        """Reserve bounded units without adding mutable queue-side state."""
        if not len(row):
            return row
        pending = torch.zeros_like(supply.item_inventory)
        for batch in self._pending.values():
            order = batch.event(EventType.ORDER) & (batch.item_id >= 0)
            pending.index_add_(
                0,
                batch.item_id[order],
                torch.ones_like(batch.item_id[order], dtype=torch.float),
            )
        units = torch.floor(25.0 * supply.item_inventory).long()
        available = (units - pending.long()).clamp_min(0)
        ordered = row[torch.argsort(events.event_id[row], stable=True)]
        keep = torch.zeros(len(ordered), device=row.device, dtype=torch.bool)
        item = events.item_id[ordered]
        for item_id in torch.unique(item, sorted=True).tolist():
            positions = torch.where(item == item_id)[0]
            keep[positions[: int(available[item_id])]] = True
        return ordered[keep]

    def _payments(
        self, events: AppEventBatch, users: HiddenUserState,
    ) -> AppEventBatch:
        order = events.event(EventType.ORDER) & (events.item_id >= 0)
        row = torch.where(order)[0]
        if not len(row):
            return AppEventBatch.empty(events.event_id.device)
        probability = torch.sigmoid(
            1.5
            + 0.65 * users.spending_power[events.user_id[row]]
            - 0.35 * torch.log1p(events.value[row])
        )
        chosen = uniform(
            events.event_id[row], 0, 1_509, self.seed,
        ) < probability
        row = row[chosen]
        delay = 1 + torch.floor(
            4.0 * uniform(events.event_id[row], 0, 1_511, self.seed)
        ).long()
        occurrence = events.event_time[row] + delay
        return self._derived(
            EventType.PAYMENT,
            events,
            row,
            event_time=occurrence,
            ingest_time=occurrence,
            value=events.value[row],
        )

    def _refunds(
        self,
        events: AppEventBatch,
        truth: HiddenCatalogTruth,
        supply: HiddenSupplyState,
    ) -> AppEventBatch:
        payment = events.event(EventType.PAYMENT) & (events.item_id >= 0)
        row = torch.where(payment)[0]
        if not len(row):
            return AppEventBatch.empty(events.event_id.device)
        item = events.item_id[row]
        merchant = events.merchant_id[row]
        probability = torch.sigmoid(
            -4.0
            + 2.0 * truth.risk[item]
            + 1.2 * (1.0 - supply.merchant_reliability[merchant])
        )
        chosen = uniform(
            events.event_id[row], 0, 1_513, self.seed,
        ) < probability
        row = row[chosen]
        delay = self.ticks_per_day + torch.floor(
            13.0 * self.ticks_per_day
            * uniform(events.event_id[row], 0, 1_517, self.seed)
        ).long()
        occurrence = events.event_time[row] + delay
        return self._derived(
            EventType.REFUND,
            events,
            row,
            event_time=occurrence,
            ingest_time=occurrence,
            value=-events.value[row].abs(),
        )

    def _pixel_conversions(
        self,
        events: AppEventBatch,
        users: HiddenUserState,
        truth: HiddenCatalogTruth,
    ) -> AppEventBatch:
        ad_click = (
            events.event(EventType.CLICK)
            & (events.content_kind == int(ContentKind.AD))
            & (events.item_id >= 0)
        )
        row = torch.where(ad_click)[0]
        if not len(row):
            return AppEventBatch.empty(events.event_id.device)
        user, item = events.user_id[row], events.item_id[row]
        probability = torch.sigmoid(
            -3.4
            + 0.9 * users.spending_power[user]
            + 1.0 * truth.quality[item]
            - 0.8 * truth.risk[item]
        )
        chosen = uniform(
            events.event_id[row], 0, 1_521, self.seed,
        ) < probability
        row = row[chosen]
        conversion_delay = 1 + torch.floor(
            self.ticks_per_day
            * uniform(events.event_id[row], 0, 1_523, self.seed)
        ).long()
        reporting_delay = torch.floor(
            2.0 * self.ticks_per_day
            * uniform(events.event_id[row], 0, 1_527, self.seed).square()
        ).long()
        occurrence = events.event_time[row] + conversion_delay
        ingest = occurrence + reporting_delay
        return self._derived(
            EventType.PIXEL_CONVERSION,
            events,
            row,
            event_time=occurrence,
            ingest_time=ingest,
            value=self.catalog.price[events.item_id[row]].clamp_min(1.0),
        )

    def _derived(
        self,
        event_type: EventType,
        source: AppEventBatch,
        row: torch.Tensor,
        *,
        event_time: torch.Tensor,
        ingest_time: torch.Tensor,
        value: torch.Tensor,
    ) -> AppEventBatch:
        item = source.item_id[row]
        order_id = torch.where(
            source.order_id[row] >= 0,
            source.order_id[row],
            source.request_id[row] * 10_000 + source.position[row],
        )
        return make_app_events(
            event_type,
            event_time=event_time,
            ingest_time=ingest_time,
            request_id=source.request_id[row],
            user_id=source.user_id[row],
            surface=source.surface[row],
            item_id=item,
            position=source.position[row],
            creator_id=source.creator_id[row],
            merchant_id=source.merchant_id[row],
            advertiser_id=source.advertiser_id[row],
            product_id=source.product_id[row],
            poi_id=source.poi_id[row],
            order_id=order_id,
            content_kind=source.content_kind[row],
            topic_id=source.topic_id[row],
            country=source.country[row],
            region=source.region[row],
            query_id=source.query_id[row],
            value=value,
            experiment_cell=source.experiment_cell[row],
            logging_probability=source.logging_probability[row],
            assignment_probability=source.assignment_probability[row],
            ordinal=source.position[row],
        )
