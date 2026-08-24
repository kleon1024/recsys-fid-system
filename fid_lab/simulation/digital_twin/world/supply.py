"""Hidden creator, merchant and advertiser agents driven by factual events."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...randomness.counter import normal, uniform
from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    ContentKind,
    EventType,
    PublishFailureReason,
    Surface,
    make_app_events,
)


@dataclass
class HiddenSupplyState:
    creator_motivation: torch.Tensor
    creator_cost: torch.Tensor
    creator_next_publish: torch.Tensor
    creator_last_publish: torch.Tensor
    creator_last_feedback: torch.Tensor
    creator_retained: torch.Tensor
    item_active: torch.Tensor
    item_creator_id: torch.Tensor
    item_product_id: torch.Tensor
    item_poi_id: torch.Tensor
    item_country: torch.Tensor
    item_region: torch.Tensor
    item_publish_time: torch.Tensor
    item_moderation_risk: torch.Tensor
    item_delete_propensity: torch.Tensor
    item_removed_reason: torch.Tensor
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
        creator_next_publish=torch.zeros_like(creators),
        creator_last_publish=torch.full_like(creators, -1),
        creator_last_feedback=torch.full_like(creators, -1),
        creator_retained=torch.ones(
            len(creators), device=device, dtype=torch.bool,
        ),
        item_active=catalog.active.clone(),
        item_creator_id=catalog.creator_id.clone(),
        item_product_id=catalog.product_id.clone(),
        item_poi_id=catalog.poi_id.clone(),
        item_country=catalog.country.clone(),
        item_region=catalog.region.clone(),
        item_publish_time=catalog.publish_time.clone(),
        item_moderation_risk=uniform(
            catalog.item_id, 0, 1_413, seed,
        ),
        item_delete_propensity=uniform(
            catalog.item_id, 0, 1_417, seed,
        ),
        item_removed_reason=torch.zeros_like(catalog.item_id),
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

    def _allocate_posts(
        self,
        creator: torch.Tensor,
        event_id: torch.Tensor,
        event_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assigned = torch.full_like(creator, -1)
        failure = torch.full_like(
            creator, int(PublishFailureReason.NO_CAPACITY),
        )
        post_kind = (
            (self.catalog.content_kind == int(ContentKind.SHORT_VIDEO))
            | (self.catalog.content_kind == int(ContentKind.PHOTO))
            | (self.catalog.content_kind == int(ContentKind.ARTICLE))
            | (self.catalog.content_kind == int(ContentKind.CARD))
        )
        available = ~self.state.item_active & post_kind
        for creator_id in torch.unique(creator).tolist():
            rows = torch.where(creator == creator_id)[0]
            rows = rows[torch.argsort(event_id[rows], stable=True)]
            if not bool(self.state.creator_retained[creator_id]):
                failure[rows] = int(PublishFailureReason.CREATOR_EXITED)
                continue
            ready = event_time[rows] >= self.state.creator_next_publish[creator_id]
            failure[rows[~ready]] = int(PublishFailureReason.CREATOR_COOLDOWN)
            rows = rows[ready]
            items = torch.where(
                available & (self.state.item_creator_id == creator_id)
            )[0]
            count = min(len(rows), len(items))
            assigned[rows[:count]] = items[:count]
            failure[rows[:count]] = 0
        return assigned, failure

    def materialize_user_posts(self, events: AppEventBatch) -> AppEventBatch:
        intent = (
            events.event(EventType.PUBLISH)
            & (events.user_id >= 0)
            & (events.surface == int(Surface.POSTING))
            & (events.source_candidate_id >= 0)
        )
        if not intent.any():
            return events
        intents = events.select(intent)
        post, failure_reason = self._allocate_posts(
            intents.creator_id, intents.event_id, intents.event_time,
        )
        fulfilled = post >= 0
        published = self._published_events(intents, post, fulfilled)
        failed = self._failed_publish_events(
            intents, fulfilled, failure_reason,
        )
        return AppEventBatch.concatenate((
            events.select(~intent), published, failed,
        ))

    def _published_events(
        self,
        intents: AppEventBatch,
        post: torch.Tensor,
        fulfilled: torch.Tensor,
    ) -> AppEventBatch:
        source = intents.source_candidate_id[fulfilled]
        item = post[fulfilled]
        return make_app_events(
            EventType.PUBLISH,
            event_time=intents.event_time[fulfilled],
            ingest_time=intents.ingest_time[fulfilled],
            request_id=intents.request_id[fulfilled],
            user_id=intents.user_id[fulfilled],
            surface=intents.surface[fulfilled],
            item_id=item,
            post_id=item,
            source_candidate_id=source,
            position=intents.position[fulfilled],
            creator_id=intents.creator_id[fulfilled],
            product_id=intents.product_id[fulfilled],
            poi_id=intents.poi_id[fulfilled],
            content_kind=self.catalog.content_kind[item],
            topic_id=self.catalog.topic_id[item],
            country=intents.country[fulfilled],
            region=intents.region[fulfilled],
            experiment_cell=intents.experiment_cell[fulfilled],
            logging_probability=intents.logging_probability[fulfilled],
            assignment_probability=intents.assignment_probability[fulfilled],
            ordinal=intents.position[fulfilled],
        )

    @staticmethod
    def _failed_publish_events(
        intents: AppEventBatch,
        fulfilled: torch.Tensor,
        failure_reason: torch.Tensor,
    ) -> AppEventBatch:
        failed = ~fulfilled
        return make_app_events(
            EventType.PUBLISH_FAILED,
            event_time=intents.event_time[failed],
            ingest_time=intents.ingest_time[failed],
            request_id=intents.request_id[failed],
            user_id=intents.user_id[failed],
            surface=intents.surface[failed],
            item_id=intents.source_candidate_id[failed],
            source_candidate_id=intents.source_candidate_id[failed],
            position=intents.position[failed],
            creator_id=intents.creator_id[failed],
            product_id=intents.product_id[failed],
            poi_id=intents.poi_id[failed],
            content_kind=intents.content_kind[failed],
            topic_id=intents.topic_id[failed],
            country=intents.country[failed],
            region=intents.region[failed],
            experiment_cell=intents.experiment_cell[failed],
            logging_probability=intents.logging_probability[failed],
            assignment_probability=intents.assignment_probability[failed],
            value=failure_reason[failed].float(),
            ordinal=intents.position[failed],
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

    def _removal_events(
        self, logical_time: int, event_type: EventType, selected: torch.Tensor,
    ) -> AppEventBatch:
        item = self.catalog.item_id[selected]
        return make_app_events(
            event_type,
            event_time=logical_time,
            request_id=self._request_id(logical_time, item),
            user_id=torch.full_like(item, -1),
            surface=torch.full_like(item, -1),
            item_id=item,
            post_id=item,
            creator_id=self.state.item_creator_id[item],
            content_kind=self.catalog.content_kind[item],
            topic_id=self.catalog.topic_id[item],
            country=self.state.item_country[item],
            region=self.state.item_region[item],
        )

    def _content_removal_events(self, logical_time: int) -> AppEventBatch:
        state = self.state
        post_kind = (
            (self.catalog.content_kind == int(ContentKind.SHORT_VIDEO))
            | (self.catalog.content_kind == int(ContentKind.PHOTO))
            | (self.catalog.content_kind == int(ContentKind.ARTICLE))
            | (self.catalog.content_kind == int(ContentKind.CARD))
        )
        eligible = state.item_active & post_kind
        moderation = eligible & (state.item_moderation_risk > 0.995)
        age = logical_time - state.item_publish_time.clamp_max(logical_time)
        deletion_score = state.item_delete_propensity * (
            1.0 - state.creator_motivation[state.item_creator_id]
        )
        deletion = (
            eligible
            & ~moderation
            & (age >= 7 * self.ticks_per_day)
            & (deletion_score > 0.995)
        )
        return AppEventBatch.concatenate((
            self._removal_events(
                logical_time, EventType.MODERATION_REMOVE, moderation,
            ),
            self._removal_events(
                logical_time, EventType.CONTENT_DELETE, deletion,
            ),
        ))

    def _creator_exit_events(self, logical_time: int) -> AppEventBatch:
        state = self.state
        creator = torch.arange(
            len(state.creator_motivation), device=self.catalog.item_id.device,
        )
        inactivity = logical_time - torch.maximum(
            state.creator_last_publish, state.creator_last_feedback,
        )
        exit_mask = (
            state.creator_retained
            & (state.creator_motivation < 0.025)
            & (inactivity >= 30 * self.ticks_per_day)
        )
        creator = creator[exit_mask]
        return make_app_events(
            EventType.CREATOR_EXIT,
            event_time=logical_time,
            request_id=self._request_id(logical_time, creator),
            user_id=torch.full_like(creator, -1),
            surface=torch.full_like(creator, -1),
            creator_id=creator,
        )

    def schedule(self, logical_time: int) -> AppEventBatch:
        return AppEventBatch.concatenate((
            self._inventory_events(logical_time),
            self._bid_events(logical_time),
            self._content_removal_events(logical_time),
            self._creator_exit_events(logical_time),
        ))

    def commit(self, events: AppEventBatch) -> None:
        state = self.state
        supply_publish = events.event(EventType.PUBLISH) & (events.post_id >= 0)
        publish_item = events.post_id[supply_publish]
        publish_creator = events.creator_id[supply_publish]
        state.item_active[publish_item] = True
        state.item_creator_id[publish_item] = publish_creator
        state.item_product_id[publish_item] = events.product_id[supply_publish]
        state.item_poi_id[publish_item] = events.poi_id[supply_publish]
        state.item_country[publish_item] = events.country[supply_publish]
        state.item_region[publish_item] = events.region[supply_publish]
        state.item_publish_time[publish_item] = events.event_time[supply_publish]
        state.item_removed_reason[publish_item] = 0
        state.creator_last_publish[publish_creator] = events.event_time[
            supply_publish
        ]
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
        moderation = events.event(EventType.MODERATION_REMOVE)
        state.item_active[events.item_id[moderation]] = False
        state.item_removed_reason[events.item_id[moderation]] = 1
        deletion = events.event(EventType.CONTENT_DELETE)
        state.item_active[events.item_id[deletion]] = False
        state.item_removed_reason[events.item_id[deletion]] = 2
        creator_exit = events.event(EventType.CREATOR_EXIT)
        state.creator_retained[events.creator_id[creator_exit]] = False
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
        touched_creator = torch.where(touched)[0]
        if len(touched_creator):
            state.creator_last_feedback[touched_creator] = int(
                events.event_time.max()
            )
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
