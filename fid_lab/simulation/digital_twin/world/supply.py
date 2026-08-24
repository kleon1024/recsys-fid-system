"""Hidden creator, merchant and advertiser agents driven by factual events."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...randomness.counter import normal, uniform
from ..catalog import PublicCatalog
from ..contracts import AppEventBatch, ContentKind, EventType, make_app_events


@dataclass
class HiddenSupplyState:
    creator_motivation: torch.Tensor
    creator_cost: torch.Tensor
    creator_next_publish: torch.Tensor
    creator_retained: torch.Tensor
    item_active: torch.Tensor
    item_inventory: torch.Tensor
    merchant_capacity: torch.Tensor
    merchant_reliability: torch.Tensor
    advertiser_budget: torch.Tensor
    advertiser_bid: torch.Tensor
    advertiser_value: torch.Tensor


def _entity_count(ids: torch.Tensor) -> int:
    return int(ids.max()) + 1 if len(ids) else 0


def build_hidden_supply(
    catalog: PublicCatalog, seed: int,
) -> HiddenSupplyState:
    device = catalog.item_id.device
    creators = torch.arange(_entity_count(catalog.creator_id), device=device)
    merchants = torch.arange(_entity_count(catalog.merchant_id), device=device)
    advertisers = torch.arange(
        _entity_count(catalog.advertiser_id), device=device,
    )
    return HiddenSupplyState(
        creator_motivation=(
            0.22 + 0.68 * uniform(creators, 0, 1_401, seed)
        ),
        creator_cost=(
            0.08 + 0.55 * uniform(creators, 0, 1_409, seed).square()
        ),
        creator_next_publish=torch.floor(
            96.0 * uniform(creators, 0, 1_411, seed)
        ).long(),
        creator_retained=torch.ones(
            len(creators), device=device, dtype=torch.bool,
        ),
        item_active=catalog.active.clone(),
        item_inventory=(
            0.15 + 0.85 * catalog.inventory
        ).clamp(0.0, 1.0),
        merchant_capacity=(
            0.35 + 0.65 * uniform(merchants, 0, 1_419, seed)
        ),
        merchant_reliability=(
            0.50 + 0.48 * uniform(merchants, 0, 1_423, seed)
        ),
        advertiser_budget=torch.exp(
            5.0 + 4.0 * uniform(advertisers, 0, 1_429, seed)
        ),
        advertiser_bid=torch.exp(
            -1.0 + 2.0 * uniform(advertisers, 0, 1_433, seed)
        ),
        advertiser_value=torch.exp(
            0.5 + 3.0 * uniform(advertisers, 0, 1_439, seed)
            + 0.15 * normal(advertisers, 0, 1_443, seed)
        ),
    )


class SupplyEcosystem:
    def __init__(
        self, catalog: PublicCatalog, environment_seed: int, ticks_per_day: int,
    ):
        self.catalog = catalog
        self.seed = environment_seed
        self.ticks_per_day = ticks_per_day
        self.state = build_hidden_supply(catalog, environment_seed)

    def _request_id(
        self, logical_time: int, item: torch.Tensor,
    ) -> torch.Tensor:
        return 4_000_000_000_000 + logical_time * len(
            self.catalog.item_id,
        ) + item

    def _next_item_by_creator(self) -> torch.Tensor:
        creators = len(self.state.creator_motivation)
        sentinel = len(self.catalog.item_id)
        result = torch.full(
            (creators,), sentinel,
            device=self.catalog.item_id.device,
            dtype=torch.long,
        )
        inactive = ~self.state.item_active
        result.scatter_reduce_(
            0,
            self.catalog.creator_id[inactive],
            self.catalog.item_id[inactive],
            reduce="amin",
            include_self=True,
        )
        return result

    def _publish_events(self, logical_time: int) -> AppEventBatch:
        state = self.state
        creator = torch.arange(
            len(state.creator_motivation),
            device=self.catalog.item_id.device,
        )
        available_item = self._next_item_by_creator()
        publish_probability = (
            0.02
            + 0.32 * state.creator_motivation
            - 0.18 * state.creator_cost
        ).clamp(0.0, 0.55)
        ready = (
            state.creator_retained
            & (state.creator_next_publish <= logical_time)
            & (available_item < len(self.catalog.item_id))
            & (
                uniform(creator, logical_time, 1_451, self.seed)
                < publish_probability
            )
        )
        creator = creator[ready]
        item = available_item[ready]
        return make_app_events(
            EventType.PUBLISH,
            event_time=logical_time,
            request_id=self._request_id(logical_time, item),
            user_id=torch.full_like(item, -1),
            surface=torch.full_like(item, -1),
            item_id=item,
            creator_id=creator,
            merchant_id=self.catalog.merchant_id[item],
            advertiser_id=self.catalog.advertiser_id[item],
            content_kind=self.catalog.content_kind[item],
            topic_id=self.catalog.topic_id[item],
            country=self.catalog.country[item],
            region=self.catalog.region[item],
        )

    def _inventory_events(self, logical_time: int) -> AppEventBatch:
        state = self.state
        item = self.catalog.item_id
        commerce_item = (
            (self.catalog.content_kind == int(ContentKind.PRODUCT))
            | (self.catalog.content_kind == int(ContentKind.POI))
        )
        merchant = self.catalog.merchant_id
        replenish = (
            state.item_active
            & commerce_item
            & (state.item_inventory < 0.28)
            & (
                uniform(item, logical_time, 1_457, self.seed)
                < 0.12 * state.merchant_reliability[merchant]
            )
        )
        item = item[replenish]
        next_inventory = (
            0.45
            + 0.5 * state.merchant_capacity[merchant[item]]
            * uniform(item, logical_time, 1_461, self.seed)
        ).clamp(0.0, 1.0)
        return make_app_events(
            EventType.INVENTORY,
            event_time=logical_time,
            request_id=self._request_id(logical_time, item),
            user_id=torch.full_like(item, -1),
            surface=torch.full_like(item, -1),
            item_id=item,
            creator_id=self.catalog.creator_id[item],
            merchant_id=self.catalog.merchant_id[item],
            content_kind=self.catalog.content_kind[item],
            value=next_inventory,
        )

    def _bid_events(self, logical_time: int) -> AppEventBatch:
        state = self.state
        advertiser = torch.arange(
            len(state.advertiser_bid), device=self.catalog.item_id.device,
        )
        bid_probability = (
            0.03 + 0.12 * (state.advertiser_budget > 1.0).float()
        )
        ready = uniform(
            advertiser, logical_time, 1_463, self.seed,
        ) < bid_probability
        advertiser = advertiser[ready]
        if not len(advertiser):
            return AppEventBatch.empty(self.catalog.item_id.device)
        ad_item = self._representative_ad_item(advertiser)
        valid = ad_item >= 0
        advertiser = advertiser[valid]
        item = ad_item[valid]
        pacing = (
            state.advertiser_budget[advertiser]
            / state.advertiser_budget.mean().clamp_min(1.0)
        ).clamp(0.1, 4.0)
        next_bid = (
            state.advertiser_value[advertiser]
            * (0.05 + 0.10 * torch.log1p(pacing))
        ).clamp(0.01, 50.0)
        return make_app_events(
            EventType.BID,
            event_time=logical_time,
            request_id=self._request_id(logical_time, item),
            user_id=torch.full_like(item, -1),
            surface=torch.full_like(item, -1),
            item_id=item,
            creator_id=self.catalog.creator_id[item],
            merchant_id=self.catalog.merchant_id[item],
            advertiser_id=advertiser,
            content_kind=self.catalog.content_kind[item],
            value=next_bid,
        )

    def _representative_ad_item(
        self, advertiser: torch.Tensor,
    ) -> torch.Tensor:
        count = len(self.state.advertiser_bid)
        sentinel = len(self.catalog.item_id)
        first = torch.full(
            (count,), sentinel,
            device=self.catalog.item_id.device,
            dtype=torch.long,
        )
        ad = (
            self.state.item_active
            & (self.catalog.content_kind == int(ContentKind.AD))
        )
        first.scatter_reduce_(
            0,
            self.catalog.advertiser_id[ad],
            self.catalog.item_id[ad],
            reduce="amin",
            include_self=True,
        )
        selected = first[advertiser]
        return torch.where(
            selected < sentinel, selected, torch.full_like(selected, -1),
        )

    def schedule(self, logical_time: int) -> AppEventBatch:
        return AppEventBatch.concatenate((
            self._publish_events(logical_time),
            self._inventory_events(logical_time),
            self._bid_events(logical_time),
        ))

    def commit(self, events: AppEventBatch) -> None:
        state = self.state
        supply_publish = events.event(EventType.PUBLISH) & (events.user_id < 0)
        publish_item = events.item_id[supply_publish]
        publish_creator = events.creator_id[supply_publish]
        state.item_active[publish_item] = True
        state.creator_motivation[publish_creator] = (
            state.creator_motivation[publish_creator]
            - 0.08 * state.creator_cost[publish_creator]
        ).clamp(0.0, 1.0)
        state.creator_next_publish[publish_creator] = (
            events.event_time[supply_publish]
            + torch.ceil(
                self.ticks_per_day
                * (0.3 + 2.5 * state.creator_cost[publish_creator])
            ).long()
        )
        inventory = events.event(EventType.INVENTORY)
        state.item_inventory[events.item_id[inventory]] = events.value[inventory]
        bid = events.event(EventType.BID)
        state.advertiser_bid[events.advertiser_id[bid]] = events.value[bid]
        self._commit_market_response(events)

    def _commit_market_response(self, events: AppEventBatch) -> None:
        state = self.state
        valid_item = events.item_id >= 0
        impression = events.event(EventType.IMPRESSION) & valid_item
        positive = (
            events.event(EventType.LONG_VIEW)
            | events.event(EventType.LIKE)
            | events.event(EventType.SHARE)
            | events.event(EventType.FOLLOW)
            | events.event(EventType.ORDER)
            | events.event(EventType.PAYMENT)
        ) & valid_item
        negative = events.event(EventType.NEGATIVE) & valid_item
        creator_gain = torch.zeros_like(state.creator_motivation)
        creator_loss = torch.zeros_like(state.creator_motivation)
        creator_gain.index_add_(
            0,
            events.creator_id[positive],
            torch.ones_like(events.creator_id[positive], dtype=torch.float),
        )
        creator_loss.index_add_(
            0,
            events.creator_id[negative],
            torch.ones_like(events.creator_id[negative], dtype=torch.float),
        )
        exposure = torch.zeros_like(state.creator_motivation)
        exposure.index_add_(
            0,
            events.creator_id[impression],
            torch.ones_like(events.creator_id[impression], dtype=torch.float),
        )
        touched = (exposure + creator_gain + creator_loss) > 0
        response_rate = creator_gain / exposure.clamp_min(1.0)
        state.creator_motivation[touched] = (
            0.995 * state.creator_motivation[touched]
            + 0.06 * response_rate[touched]
            - 0.04 * creator_loss[touched] / exposure[touched].clamp_min(1.0)
        ).clamp(0.0, 1.0)
        order = events.event(EventType.ORDER) & valid_item
        order_item = events.item_id[order]
        order_count = torch.zeros_like(state.item_inventory)
        order_count.index_add_(
            0, order_item, torch.ones_like(order_item, dtype=torch.float),
        )
        state.item_inventory.sub_(0.04 * order_count).clamp_min_(0.0)
        refund = events.event(EventType.REFUND) & valid_item
        refund_item = events.item_id[refund]
        refund_count = torch.zeros_like(state.item_inventory)
        refund_count.index_add_(
            0, refund_item, torch.ones_like(refund_item, dtype=torch.float),
        )
        state.item_inventory.add_(0.02 * refund_count).clamp_max_(1.0)
        refund_merchant = events.merchant_id[refund]
        reliability_loss = torch.zeros_like(state.merchant_reliability)
        reliability_loss.index_add_(
            0,
            refund_merchant,
            torch.ones_like(refund_merchant, dtype=torch.float),
        )
        state.merchant_reliability.sub_(
            0.002 * reliability_loss
        ).clamp_(0.05, 0.999)
        ad_impression = impression & (
            events.content_kind == int(ContentKind.AD)
        )
        ad = events.advertiser_id[ad_impression]
        if len(ad):
            spend = state.advertiser_bid[ad]
            state.advertiser_budget.index_add_(0, ad, -spend)
            state.advertiser_budget.clamp_min_(0.0)
        pixel = events.event(EventType.PIXEL_CONVERSION)
        pixel_advertiser = events.advertiser_id[pixel]
        if len(pixel_advertiser):
            observed_value = torch.zeros_like(state.advertiser_value)
            conversion_count = torch.zeros_like(state.advertiser_value)
            observed_value.index_add_(
                0, pixel_advertiser, events.value[pixel].clamp_min(0.0),
            )
            conversion_count.index_add_(
                0,
                pixel_advertiser,
                torch.ones_like(pixel_advertiser, dtype=torch.float),
            )
            converted = conversion_count > 0
            state.advertiser_value[converted] = (
                0.97 * state.advertiser_value[converted]
                + 0.03 * observed_value[converted] / conversion_count[converted]
            )
