"""Event-only facade for the private user ecosystem."""

from __future__ import annotations

from dataclasses import fields

import torch

from ...randomness.counter import uniform
from ..catalog import PublicCatalog
from ..contracts import (
    AppEventBatch,
    EventType,
    RenderedSlateBatch,
    Surface,
    make_app_events,
)
from .authority import FormulaResponseAuthority, ResponseAuthority
from .dynamics.calendar import (
    CALENDAR_VERSION,
    arrival_hazard,
    sample_return_outcome,
)
from .dynamics.population import POPULATION_VERSION
from .delayed import DelayedOutcomeQueue
from .state import (
    HiddenUserState,
    UserWorldConfig,
    UserWorldSnapshot,
    build_hidden_catalog_truth,
    build_hidden_users,
    topic_prototypes,
)
from .supply import SupplyEcosystem
from .dynamics.trends import TREND_VERSION, TrendProcess


class UserEcosystemWorld:
    """Owns hidden state; the platform sees only emitted app events."""

    def __init__(
        self,
        config: UserWorldConfig,
        catalog: PublicCatalog,
        response_authority: ResponseAuthority | None = None,
    ):
        if catalog.content_embedding.shape[1] != config.embedding_dim:
            raise ValueError("catalog and world embedding dimensions differ")
        if int(catalog.topic_id.max()) >= config.topics:
            raise ValueError("catalog topic IDs exceed world topic count")
        self.config = config
        self.catalog = catalog
        self.response_authority = (
            FormulaResponseAuthority()
            if response_authority is None else response_authority
        )
        self.users = build_hidden_users(config, catalog)
        self.catalog_truth = build_hidden_catalog_truth(
            catalog, config.environment_seed,
        )
        self._topic_prototypes = topic_prototypes(catalog, config.topics)
        self.trends = TrendProcess(
            config.countries * config.regions_per_country,
            config.topics,
            config.environment_seed,
            catalog.item_id.device,
        )
        self.supply = SupplyEcosystem(
            catalog, config.environment_seed, config.ticks_per_day,
        )
        self.delayed = DelayedOutcomeQueue(
            catalog, config.environment_seed, config.ticks_per_day,
        )

    @property
    def max_reporting_lag(self) -> int:
        return 2 * self.config.ticks_per_day

    def manifest(self) -> dict[str, str]:
        return {
            "population": POPULATION_VERSION,
            "calendar_survival": CALENDAR_VERSION,
            "trends": TREND_VERSION,
            "response": self.response_authority.version,
        }

    def snapshot(self) -> UserWorldSnapshot:
        return UserWorldSnapshot(
            self.users.clone(),
            self.catalog_truth,
            self.config.ticks_per_day,
            self.config.environment_seed,
            self.supply.state.item_creator_id.clone(),
            self.supply.state.item_product_id.clone(),
            self.supply.state.item_poi_id.clone(),
            self.supply.state.item_country.clone(),
            self.supply.state.item_region.clone(),
            self.supply.state.item_publish_time.clone(),
            self.trends.snapshot(),
        )

    def _surface(self, user: torch.Tensor, logical_time: int) -> torch.Tensor:
        state = self.users
        local_hour = (
            logical_time * 24.0 / self.config.ticks_per_day
            + state.timezone_offset[user].float()
        ).remainder(24.0)
        logits = torch.log(state.surface_intent[user].clamp_min(1e-5))
        logits[:, int(Surface.FEED)] += 0.35
        dinner = torch.exp(-((local_hour - 19.0) / 2.5).square())
        logits[:, int(Surface.LOCAL)] += 0.9 * dinner
        logits[:, int(Surface.COMMERCE)] += 0.45 * dinner
        logits[:, int(Surface.LIVE)] += 0.55 * torch.exp(
            -((local_hour - 21.0) / 3.0).square()
        )
        draw = uniform(
            user,
            logical_time,
            1_301,
            self.config.environment_seed,
            len(Surface),
        ).clamp(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(draw))
        return torch.argmax(logits + 0.42 * gumbel, dim=1)

    def schedule(self, logical_time: int) -> AppEventBatch:
        self.trends.advance(logical_time)
        state = self.users
        user = state.user_id
        registration = (~state.registered) & (state.signup_time <= logical_time)
        eligible = state.registered | registration
        arrival_probability = arrival_hazard(
            state, logical_time, self.config.ticks_per_day,
        )
        reactivation = (
            state.churned
            & (state.reactivation_time <= logical_time)
        )
        arrival = (
            eligible
            & ~state.active
            & ~state.churned
            & (state.next_return_time <= logical_time)
            & (
                uniform(
                    user,
                    logical_time,
                    1_307,
                    self.config.environment_seed,
                ) < arrival_probability
            )
        )
        arrival |= registration | reactivation
        requesting = state.active | arrival
        request_user = user[requesting]
        request_id = logical_time * self.config.users + request_user + 1
        surface = self._surface(request_user, logical_time)
        request_time = torch.full_like(request_user, logical_time)
        country = state.country[request_user]
        region = state.region[request_user]
        registration_user = user[registration]
        registration_request = (
            logical_time * self.config.users + registration_user + 1
        )
        registration_events = make_app_events(
            EventType.REGISTRATION,
            event_time=logical_time,
            request_id=registration_request,
            user_id=registration_user,
            surface=torch.full_like(registration_user, -1),
            creator_id=state.creator_id[registration_user],
            country=state.country[registration_user],
            region=state.region[registration_user],
        )
        arrival_row = arrival[request_user]
        session_events = make_app_events(
            EventType.SESSION_START,
            event_time=request_time[arrival_row],
            request_id=request_id[arrival_row],
            user_id=request_user[arrival_row],
            surface=surface[arrival_row],
            country=country[arrival_row],
            region=region[arrival_row],
        )
        surface_events = make_app_events(
            EventType.SURFACE_ENTRY,
            event_time=request_time,
            request_id=request_id,
            user_id=request_user,
            surface=surface,
            country=country,
            region=region,
        )
        query_row = surface == int(Surface.SEARCH)
        query_user = request_user[query_row]
        query_topic = torch.argmax(
            self.users.short_interest[query_user]
            @ self._topic_prototypes.T,
            dim=1,
        )
        query_events = make_app_events(
            EventType.QUERY,
            event_time=request_time[query_row],
            request_id=request_id[query_row],
            user_id=query_user,
            surface=surface[query_row],
            country=country[query_row],
            region=region[query_row],
            query_id=request_id[query_row] * 31 + query_topic,
            topic_id=query_topic,
        )
        user_events = AppEventBatch.concatenate((
            registration_events,
            session_events,
            surface_events,
            query_events,
        ))
        return AppEventBatch.concatenate((
            user_events,
            self.supply.schedule(logical_time),
            self.delayed.due(logical_time),
        ))

    def respond(
        self,
        snapshot: UserWorldSnapshot,
        slate: RenderedSlateBatch,
    ) -> AppEventBatch:
        events = self.response_authority.respond(
            snapshot,
            self.catalog,
            slate,
            self.config.environment_seed,
        )
        return self.supply.materialize_user_posts(events)

    def commit(self, events: AppEventBatch) -> None:
        if not len(events.event_id):
            return
        self.delayed.acknowledge(events)
        self._commit_lifecycle(events)
        self._commit_engagement(events)
        self._commit_sequence(events)
        self.supply.commit(events)
        self.trends.commit(events)
        self.delayed.schedule_from(
            events,
            self.users,
            self.catalog_truth,
            self.supply.state,
        )

    def _commit_lifecycle(self, events: AppEventBatch) -> None:
        state = self.users
        registration = events.event(EventType.REGISTRATION)
        state.registered[events.user_id[registration]] = True
        start = events.event(EventType.SESSION_START)
        start_user = events.user_id[start]
        state.active[start_user] = True
        state.churned[start_user] = False
        state.reactivation_time[start_user] = torch.iinfo(torch.long).max // 4
        state.session_depth[start_user] = 0
        state.session_count[start_user] += 1
        state.fatigue[start_user] *= 0.72
        self._drift_short_interest(start_user, events.event_time[start])
        entry = events.event(EventType.SURFACE_ENTRY)
        state.session_depth[events.user_id[entry]] += 1
        end = events.event(EventType.SESSION_END)
        end_user = events.user_id[end]
        if not len(end_user):
            return
        state.active[end_user] = False
        selected_users = HiddenUserState(**{
            field.name: getattr(state, field.name)[end_user]
            for field in fields(HiddenUserState)
        })
        outcome = sample_return_outcome(
            selected_users,
            events.event_id[end],
            events.event_time[end],
            self.config.ticks_per_day,
            self.config.environment_seed,
        )
        state.churned[end_user] = outcome.churned
        state.reactivation_time[end_user] = outcome.reactivation_time
        state.next_return_time[end_user] = (
            events.event_time[end] + outcome.delay_ticks
        )

    def _drift_short_interest(
        self, user: torch.Tensor, event_time: torch.Tensor,
    ) -> None:
        if not len(user):
            return
        state = self.users
        del event_time
        trend_topic = self.trends.top_topic(state.region[user])
        strength = (
            0.025
            + 0.055 * state.novelty[user]
            + 0.025 * (1.0 - state.habit[user])
        )[:, None]
        state.short_interest[user] = torch.nn.functional.normalize(
            (1.0 - strength) * state.short_interest[user]
            + strength * self._topic_prototypes[trend_topic],
            dim=1,
        )

    def _commit_engagement(self, events: AppEventBatch) -> None:
        state = self.users
        users = len(state.user_id)
        valid_user = events.user_id >= 0
        positive_weight = torch.zeros(len(events.event_id), device=events.event_id.device)
        weights = {
            EventType.LONG_VIEW: 0.45,
            EventType.COMPLETE: 0.70,
            EventType.LIKE: 0.35,
            EventType.COMMENT: 0.45,
            EventType.SHARE: 0.65,
            EventType.FOLLOW: 0.80,
            EventType.CLICK: 0.22,
            EventType.FAVORITE: 0.55,
            EventType.ADD_CART: 0.75,
            EventType.ORDER: 1.15,
            EventType.PAYMENT: 1.40,
            EventType.PUBLISH: 1.05,
        }
        for event_type, weight in weights.items():
            positive_weight[events.event(event_type)] = weight
        negative_weight = (
            events.event(EventType.NEGATIVE).float()
            + 0.7 * events.event(EventType.REFUND).float()
        )
        dwell = events.event(EventType.DWELL)
        dwell_value = torch.zeros_like(positive_weight)
        dwell_value[dwell] = torch.log1p(
            events.duration_ms[dwell].float().clamp_min(0.0) / 1_000.0,
        )
        examine = events.event(EventType.EXAMINE).float()
        positive = torch.zeros(users, device=events.event_id.device)
        negative = torch.zeros_like(positive)
        dwell_by_user = torch.zeros_like(positive)
        examined = torch.zeros_like(positive)
        event_user = events.user_id[valid_user]
        positive.scatter_add_(0, event_user, positive_weight[valid_user])
        negative.scatter_add_(0, event_user, negative_weight[valid_user])
        dwell_by_user.scatter_add_(0, event_user, dwell_value[valid_user])
        examined.scatter_add_(0, event_user, examine[valid_user])
        touched = (positive + negative + dwell_by_user + examined) > 0
        next_satisfaction = (
            0.992 * state.satisfaction
            + 0.025 * positive.clamp_max(3.0)
            + 0.008 * dwell_by_user.clamp_max(8.0)
            - 0.10 * negative.clamp_max(2.0)
        ).clamp(0.0, 1.0)
        next_fatigue = (
            0.975 * state.fatigue
            + 0.012 * examined.clamp_max(12.0)
            + 0.08 * negative.clamp_max(2.0)
            - 0.012 * positive.clamp_max(3.0)
        ).clamp(0.0, 1.0)
        state.satisfaction[touched] = next_satisfaction[touched]
        state.fatigue[touched] = next_fatigue[touched]
        state.habit[touched] = (
            0.996 * state.habit[touched]
            + 0.004 * state.satisfaction[touched]
        ).clamp(0.01, 0.99)
        self._commit_interest(events, positive_weight)

    def _commit_interest(
        self, events: AppEventBatch, positive_weight: torch.Tensor,
    ) -> None:
        state = self.users
        usable = (
            (positive_weight > 0)
            & (events.item_id >= 0)
            & (events.user_id >= 0)
        )
        if not usable.any():
            return
        event_user = events.user_id[usable]
        weight = positive_weight[usable]
        embedding = self.catalog_truth.semantic_embedding[events.item_id[usable]]
        aggregate = torch.zeros_like(state.short_interest)
        total = torch.zeros(len(state.user_id), device=events.event_id.device)
        aggregate.index_add_(0, event_user, weight[:, None] * embedding)
        total.index_add_(0, event_user, weight)
        touched = total > 0
        target = torch.nn.functional.normalize(
            aggregate[touched] / total[touched, None], dim=1,
        )
        rate = (0.035 + 0.10 * total[touched].clamp_max(2.0))[:, None]
        state.short_interest[touched] = torch.nn.functional.normalize(
            (1.0 - rate) * state.short_interest[touched] + rate * target,
            dim=1,
        )

    def _commit_sequence(self, events: AppEventBatch) -> None:
        impression = events.event(EventType.IMPRESSION)
        if not impression.any():
            return
        request_ids = torch.unique(events.request_id[impression], sorted=True)
        request_location = torch.searchsorted(request_ids, events.request_id)
        valid_request = (
            (request_location < len(request_ids))
            & (request_ids[request_location.clamp_max(len(request_ids) - 1)]
               == events.request_id)
        )
        rows = len(request_ids)
        event_row = request_location[valid_request]
        request_user = torch.full(
            (rows,), -1, device=events.event_id.device, dtype=torch.long,
        )
        impression_row = request_location[impression]
        request_user.scatter_(0, impression_row, events.user_id[impression])
        token = torch.zeros(rows, 8, device=events.event_id.device)
        topic = torch.zeros(rows, device=events.event_id.device)
        first_position = impression & (events.position == 0)
        topic.scatter_(
            0,
            request_location[first_position],
            events.topic_id[first_position].float() / max(self.config.topics - 1, 1),
        )
        token[:, 0] = topic
        dwell = events.event(EventType.DWELL) & valid_request
        token[:, 1].index_add_(
            0,
            request_location[dwell],
            torch.log1p(events.duration_ms[dwell].float() / 1_000.0)
            / torch.log(torch.tensor(181.0, device=events.event_id.device)),
        )
        event_channels = (
            EventType.LONG_VIEW,
            EventType.COMPLETE,
            EventType.LIKE,
            EventType.NEGATIVE,
            EventType.CLICK,
            EventType.PAYMENT,
        )
        for channel, event_type in enumerate(event_channels, start=2):
            selected = events.event(event_type) & valid_request
            token[:, channel].index_add_(
                0,
                request_location[selected],
                torch.ones_like(events.value[selected]),
            )
        token[:, 1:] = token[:, 1:].clamp(0.0, 1.0)
        valid_user = request_user >= 0
        users = request_user[valid_user]
        sequence = self.users.behavior_sequence[users]
        sequence = torch.roll(sequence, shifts=-1, dims=1)
        sequence[:, -1] = token[valid_user].to(sequence.dtype)
        self.users.behavior_sequence[users] = sequence
