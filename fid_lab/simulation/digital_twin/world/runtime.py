"""Event-only facade for the private user ecosystem."""

from __future__ import annotations

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
from .behavior import response_events
from .state import (
    UserWorldConfig,
    UserWorldSnapshot,
    build_hidden_catalog_truth,
    build_hidden_users,
    topic_prototypes,
)
from .supply import SupplyEcosystem


class UserEcosystemWorld:
    """Owns hidden state; the platform sees only emitted app events."""

    def __init__(self, config: UserWorldConfig, catalog: PublicCatalog):
        if catalog.content_embedding.shape[1] != config.embedding_dim:
            raise ValueError("catalog and world embedding dimensions differ")
        if int(catalog.topic_id.max()) >= config.topics:
            raise ValueError("catalog topic IDs exceed world topic count")
        self.config = config
        self.catalog = catalog
        self.users = build_hidden_users(config, catalog)
        self.catalog_truth = build_hidden_catalog_truth(
            catalog, config.environment_seed,
        )
        self._topic_prototypes = topic_prototypes(catalog, config.topics)
        self.supply = SupplyEcosystem(
            catalog, config.environment_seed, config.ticks_per_day,
        )

    def snapshot(self) -> UserWorldSnapshot:
        return UserWorldSnapshot(self.users.clone(), self.catalog_truth)

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
        state = self.users
        user = state.user_id
        registration = (~state.registered) & (state.signup_time <= logical_time)
        eligible = state.registered | registration
        local_hour = (
            logical_time * 24.0 / self.config.ticks_per_day
            + state.timezone_offset.float()
        ).remainder(24.0)
        circadian = (
            0.25
            + 0.45 * torch.exp(-((local_hour - 12.0) / 4.5).square())
            + 0.70 * torch.exp(-((local_hour - 21.0) / 3.5).square())
        )
        arrival_probability = (
            state.activity
            * circadian
            * (0.45 + 0.55 * state.habit)
            * (0.55 + 0.45 * state.satisfaction)
            / self.config.ticks_per_day
        ).clamp(0.0, 0.45)
        arrival = (
            eligible
            & ~state.active
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
        arrival |= registration
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
            user_events, self.supply.schedule(logical_time),
        ))

    def respond(
        self,
        snapshot: UserWorldSnapshot,
        slate: RenderedSlateBatch,
    ) -> AppEventBatch:
        return response_events(
            snapshot,
            self.catalog,
            slate,
            self.config.environment_seed,
        )

    def commit(self, events: AppEventBatch) -> None:
        if not len(events.event_id):
            return
        self._commit_lifecycle(events)
        self._commit_engagement(events)
        self.supply.commit(events)

    def _commit_lifecycle(self, events: AppEventBatch) -> None:
        state = self.users
        registration = events.event(EventType.REGISTRATION)
        state.registered[events.user_id[registration]] = True
        start = events.event(EventType.SESSION_START)
        start_user = events.user_id[start]
        state.active[start_user] = True
        state.session_depth[start_user] = 0
        state.session_count[start_user] += 1
        state.fatigue[start_user] *= 0.72
        entry = events.event(EventType.SURFACE_ENTRY)
        state.session_depth[events.user_id[entry]] += 1
        end = events.event(EventType.SESSION_END)
        end_user = events.user_id[end]
        if not len(end_user):
            return
        state.active[end_user] = False
        return_noise = uniform(
            events.event_id[end],
            0,
            1_337,
            self.config.environment_seed,
        )
        expected_days = torch.exp(
            1.4
            + 1.8 * state.fatigue[end_user]
            - 1.5 * state.satisfaction[end_user]
            - 1.2 * state.habit[end_user]
            + 0.35 * (return_noise - 0.5)
        ).clamp(0.01, 45.0)
        delay = torch.ceil(
            expected_days * self.config.ticks_per_day,
        ).long().clamp_min(1)
        state.next_return_time[end_user] = events.event_time[end] + delay

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
        negative_weight = events.event(EventType.NEGATIVE).float()
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
