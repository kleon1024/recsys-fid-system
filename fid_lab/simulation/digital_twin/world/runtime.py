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
from ..semantics import SEMANTIC_SIGNAL_VERSION
from .authority import FormulaResponseAuthority, ResponseAuthority
from .dynamics.calendar import (
    CALENDAR_VERSION,
    arrival_hazard,
    sample_return_outcome,
)
from .dynamics.population import POPULATION_VERSION
from .dynamics.growth import ACQUISITION_VERSION, GrowthProcess
from .dynamics.needs import (
    NEED_DYNAMICS_VERSION,
    NeedKind,
)
from .dynamics.lifecycle import (
    LIFECYCLE_DYNAMICS_VERSION,
    advance_latent_user_state,
    commit_need_and_activation,
    commit_session_start,
    quantize_dynamic_state,
    update_lifecycle_stage,
)
from .delayed import DelayedOutcomeQueue
from .state import (
    HiddenUserState,
    UserWorldConfig,
    UserWorldSnapshot,
    build_hidden_catalog_truth,
    build_hidden_users,
    topic_prototypes,
)
from .supply import SUPPLY_DYNAMICS_VERSION, SupplyEcosystem
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
        self.growth = GrowthProcess(
            config.countries,
            config.environment_seed,
            catalog.item_id.device,
        )
        self.supply = SupplyEcosystem(
            catalog,
            config.environment_seed,
            config.ticks_per_day,
            config.background_posts_per_day,
        )
        self.delayed = DelayedOutcomeQueue(
            catalog, config.environment_seed, config.ticks_per_day,
        )

    @property
    def max_reporting_lag(self) -> int:
        return 2 * self.config.ticks_per_day

    def manifest(self) -> dict[str, object]:
        return {
            "semantic_signal": SEMANTIC_SIGNAL_VERSION,
            "population": POPULATION_VERSION,
            "calendar_survival": CALENDAR_VERSION,
            "trends": TREND_VERSION,
            "acquisition": ACQUISITION_VERSION,
            "needs": NEED_DYNAMICS_VERSION,
            "lifecycle": LIFECYCLE_DYNAMICS_VERSION,
            "response": self.response_authority.version,
            "response_authority": self.response_authority.manifest(),
            "supply": {
                "version": SUPPLY_DYNAMICS_VERSION,
                "background_posts_per_day": self.config.background_posts_per_day,
            },
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

    def view(self) -> UserWorldSnapshot:
        """Read-only-by-contract response view valid for one atomic tick."""
        state = self.supply.state
        return UserWorldSnapshot(
            self.users,
            self.catalog_truth,
            self.config.ticks_per_day,
            self.config.environment_seed,
            state.item_creator_id,
            state.item_product_id,
            state.item_poi_id,
            state.item_country,
            state.item_region,
            state.item_publish_time,
            self.trends.state.strength,
        )

    def _surface(self, user: torch.Tensor, logical_time: int) -> torch.Tensor:
        state = self.users
        local_hour = (
            logical_time * 24.0 / self.config.ticks_per_day
            + state.timezone_offset[user].float()
        ).remainder(24.0)
        logits = torch.log(state.surface_intent[user].clamp_min(1e-5))
        logits[:, int(Surface.FEED)] += 0.35
        if self.config.initialization_mode == "equilibrium":
            need = state.need_kind[user]
            strength = state.need_strength[user]
            logits[:, int(Surface.FEED)] += strength * (
                (need == int(NeedKind.ENTERTAINMENT)).float()
                + 0.55 * (need == int(NeedKind.SOCIAL)).float()
            )
            logits[:, int(Surface.SEARCH)] += strength * (
                1.15 * (need == int(NeedKind.INFORMATION)).float()
                + 0.75 * (need == int(NeedKind.LOCAL)).float()
                + 0.65 * (need == int(NeedKind.COMMERCE)).float()
            )
            logits[:, int(Surface.LOCAL)] += 1.25 * strength * (
                need == int(NeedKind.LOCAL)
            ).float()
            logits[:, int(Surface.COMMERCE)] += 1.20 * strength * (
                need == int(NeedKind.COMMERCE)
            ).float()
            logits[:, int(Surface.POSTING)] += 1.30 * strength * (
                need == int(NeedKind.CREATION)
            ).float()
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

    def _scheduled_surface(
        self, user: torch.Tensor, logical_time: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.users
        surface = self._surface(user, logical_time)
        followup_search = state.search_followup_topic[user] >= 0
        post_search_feed = state.post_search_feed_pending[user] & ~followup_search
        surface = torch.where(
            followup_search,
            torch.full_like(surface, int(Surface.SEARCH)),
            surface,
        )
        surface = torch.where(
            post_search_feed,
            torch.full_like(surface, int(Surface.FEED)),
            surface,
        )
        return surface, followup_search, post_search_feed

    def schedule(self, logical_time: int) -> AppEventBatch:
        self.trends.advance(logical_time)
        self.growth.advance(logical_time, self.config.ticks_per_day)
        state = self.users
        user = state.user_id
        registration = advance_latent_user_state(
            state, self.growth, logical_time, self.config,
        )
        eligible = state.registered | registration
        arrival_probability = arrival_hazard(
            state,
            logical_time,
            self.config.ticks_per_day,
            self.config.arrival_intensity,
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
        surface, _, post_search_feed = self._scheduled_surface(
            request_user, logical_time,
        )
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
            query_id=torch.where(
                post_search_feed,
                state.last_search_query_id[request_user],
                torch.full_like(request_user, -1),
            ),
        )
        query_row = surface == int(Surface.SEARCH)
        query_user = request_user[query_row]
        query_topic = torch.argmax(
            self.users.short_interest[query_user]
            @ self._topic_prototypes.T,
            dim=1,
        )
        pending_topic = state.search_followup_topic[query_user]
        query_topic = torch.where(
            pending_topic >= 0, pending_topic, query_topic,
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
        state.search_followup_topic[query_user] = -1
        state.post_search_feed_pending[request_user[post_search_feed]] = False
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
        events = self.supply.materialize_user_posts(events)
        return self.supply.materialize_ad_spend(events)

    def commit(self, events: AppEventBatch) -> None:
        if not len(events.event_id):
            return
        self.delayed.acknowledge(events)
        self._commit_session_open(events)
        self._commit_engagement(events)
        self._commit_exposure_memory(events)
        self._commit_sequence(events)
        self._commit_search_state(events)
        self._commit_session_close(events)
        self.supply.commit(events)
        self.trends.commit(events)
        self.growth.commit(events)
        self.delayed.schedule_from(
            events,
            self.users,
            self.catalog_truth,
            self.supply.state,
        )
        quantize_dynamic_state(self.users)

    def _commit_search_state(self, events: AppEventBatch) -> None:
        state = self.users
        query = events.event(EventType.QUERY) & (events.user_id >= 0)
        state.last_search_query_id[events.user_id[query]] = events.query_id[query]
        reformulate = events.event(EventType.SEARCH_REFORMULATE) & (
            events.user_id >= 0
        )
        reform_user = events.user_id[reformulate]
        state.search_followup_topic[reform_user] = events.topic_id[reformulate]
        state.search_reformulation_depth[reform_user] += 1
        abandon = events.event(EventType.SEARCH_ABANDON) & (events.user_id >= 0)
        abandon_user = events.user_id[abandon]
        state.search_followup_topic[abandon_user] = -1
        state.search_reformulation_depth[abandon_user] = 0
        success = events.event(EventType.SEARCH_SUCCESS) & (events.user_id >= 0)
        if not success.any():
            return
        success_user = events.user_id[success]
        state.post_search_feed_pending[success_user] = True
        state.search_followup_topic[success_user] = -1
        state.search_reformulation_depth[success_user] = 0

    def _commit_session_open(self, events: AppEventBatch) -> None:
        state = self.users
        registration = events.event(EventType.REGISTRATION)
        registration_user = events.user_id[registration]
        state.registered[registration_user] = True
        state.lifecycle_stage[registration_user] = 1
        start = events.event(EventType.SESSION_START)
        start_user = events.user_id[start]
        state.active[start_user] = True
        state.churned[start_user] = False
        state.reactivation_time[start_user] = torch.iinfo(torch.long).max // 4
        state.session_depth[start_user] = 0
        state.session_count[start_user] += 1
        commit_session_start(
            state,
            start_user,
            events.event_time[start],
            self.config.ticks_per_day,
        )
        state.fatigue[start_user] *= 0.72
        state.disappointment[start_user] *= 0.82
        self._drift_short_interest(start_user, events.event_time[start])
        entry = events.event(EventType.SURFACE_ENTRY)
        state.session_depth[events.user_id[entry]] += 1

    def _commit_session_close(self, events: AppEventBatch) -> None:
        state = self.users
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
        state.search_followup_topic[end_user] = -1
        state.search_reformulation_depth[end_user] = 0
        state.post_search_feed_pending[end_user] = False
        update_lifecycle_stage(state, end_user)

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
        disappointment, repeat_pressure, experience_user = (
            self._experience_outcomes(events)
        )
        touched = (
            (positive + negative + dwell_by_user + examined) > 0
        ) | experience_user
        next_satisfaction = (
            0.992 * state.satisfaction
            + 0.025 * positive.clamp_max(3.0)
            + 0.008 * dwell_by_user.clamp_max(8.0)
            - 0.10 * negative.clamp_max(2.0)
            - 0.09 * disappointment
            - 0.12 * repeat_pressure
        ).clamp(0.0, 1.0)
        next_fatigue = (
            0.975 * state.fatigue
            + 0.012 * examined.clamp_max(12.0)
            + 0.08 * negative.clamp_max(2.0)
            - 0.012 * positive.clamp_max(3.0)
            + 0.10 * disappointment
            + 0.16 * repeat_pressure
        ).clamp(0.0, 1.0)
        state.satisfaction[touched] = next_satisfaction[touched]
        state.fatigue[touched] = next_fatigue[touched]
        state.habit[touched] = (
            0.996 * state.habit[touched]
            + 0.004 * state.satisfaction[touched]
        ).clamp(0.01, 0.99)
        state.disappointment[experience_user] = (
            0.82 * state.disappointment[experience_user]
            + 0.18 * disappointment[experience_user]
        ).clamp(0.0, 1.0)
        commit_need_and_activation(
            state,
            touched,
            positive,
            negative,
            dwell_by_user,
            disappointment,
            repeat_pressure,
        )
        self._commit_interest(events, positive_weight)

    def _experience_outcomes(
        self, events: AppEventBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.users
        users = len(state.user_id)
        zero = torch.zeros(users, device=events.event_id.device)
        impression = (
            events.event(EventType.IMPRESSION)
            & (events.user_id >= 0)
            & (events.item_id >= 0)
            & (events.surface == int(Surface.FEED))
        )
        if not impression.any():
            return zero, zero.clone(), zero.bool()
        request_ids = torch.unique(events.request_id[impression], sorted=True)
        location = torch.searchsorted(request_ids, events.request_id)
        aligned = (
            (location < len(request_ids))
            & (request_ids[location.clamp_max(len(request_ids) - 1)]
               == events.request_id)
        )
        request_user = torch.full(
            (len(request_ids),), -1,
            device=events.event_id.device,
            dtype=torch.long,
        )
        request_user.scatter_(
            0, location[impression], events.user_id[impression],
        )
        examined = torch.zeros(len(request_ids), device=events.event_id.device)
        examined.index_add_(
            0,
            location[events.event(EventType.EXAMINE) & aligned],
            torch.ones_like(events.value[events.event(EventType.EXAMINE) & aligned]),
        )
        success_event = (
            events.event(EventType.PLAY_3S)
            | events.event(EventType.LONG_VIEW)
            | events.event(EventType.LIKE)
            | events.event(EventType.SHARE)
            | events.event(EventType.FOLLOW)
        ) & aligned
        success = torch.zeros_like(examined)
        success.index_add_(
            0, location[success_event], torch.ones_like(events.value[success_event]),
        )
        disappointed_request = ((examined > 0) & (success == 0)).float()
        user_requests = torch.zeros_like(zero)
        user_disappointment = torch.zeros_like(zero)
        valid_request = request_user >= 0
        user_requests.index_add_(
            0, request_user[valid_request], torch.ones_like(request_user[valid_request]).float(),
        )
        user_disappointment.index_add_(
            0, request_user[valid_request], disappointed_request[valid_request],
        )
        impression_user = events.user_id[impression]
        history = state.exposure_item[impression_user]
        repeated = (
            events.item_id[impression, None] == history
        ).any(dim=1).float()
        repeat_count = torch.zeros_like(zero)
        impression_count = torch.zeros_like(zero)
        repeat_count.index_add_(0, impression_user, repeated)
        impression_count.index_add_(
            0, impression_user, torch.ones_like(impression_user).float(),
        )
        touched = user_requests > 0
        return (
            user_disappointment / user_requests.clamp_min(1.0),
            repeat_count / impression_count.clamp_min(1.0),
            touched,
        )

    def rebuild_experience(self, batches: tuple[AppEventBatch, ...]) -> None:
        """Backfill newly introduced hidden experience memory at a DGP boundary."""
        state = self.users
        state.exposure_item.fill_(-1)
        state.exposure_creator.fill_(-1)
        state.exposure_topic.fill_(-1)
        state.exposure_time.fill_(-1)
        state.exposure_positive.zero_()
        state.exposure_cursor.zero_()
        state.disappointment.zero_()
        for events in batches:
            disappointment, _, touched = self._experience_outcomes(events)
            state.disappointment[touched] = (
                0.82 * state.disappointment[touched]
                + 0.18 * disappointment[touched]
            )
            self._commit_exposure_memory(events)

    def _commit_exposure_memory(self, events: AppEventBatch) -> None:
        impression = (
            events.event(EventType.IMPRESSION)
            & (events.user_id >= 0)
            & (events.item_id >= 0)
        )
        if not impression.any():
            return
        positive_event = (
            events.event(EventType.PLAY_3S)
            | events.event(EventType.LONG_VIEW)
            | events.event(EventType.COMPLETE)
            | events.event(EventType.LIKE)
            | events.event(EventType.SHARE)
            | events.event(EventType.FOLLOW)
            | events.event(EventType.CLICK)
        ) & (events.position >= 0)
        event_key = events.request_id * 64 + events.position.clamp_min(0)
        positive_key = event_key[positive_event]
        positive = torch.isin(event_key[impression], positive_key)
        user = events.user_id[impression]
        item = events.item_id[impression]
        creator = events.creator_id[impression]
        topic = events.topic_id[impression]
        event_time = events.event_time[impression]
        time_order = torch.argsort(event_time, stable=True)
        user_order = torch.argsort(user[time_order], stable=True)
        order = time_order[user_order]
        user, item = user[order], item[order]
        creator, topic = creator[order], topic[order]
        event_time, positive = event_time[order], positive[order]
        new_user = torch.ones_like(user, dtype=torch.bool)
        new_user[1:] = user[1:] != user[:-1]
        starts = torch.where(new_user)[0]
        group = torch.cumsum(new_user.long(), dim=0) - 1
        within = torch.arange(len(user), device=user.device) - starts[group]
        ends = torch.cat((
            starts[1:], torch.tensor([len(user)], device=user.device),
        ))
        group_size = (ends - starts)[group]
        width = self.users.exposure_item.shape[1]
        keep = within >= (group_size - width).clamp_min(0)
        slot = torch.remainder(
            self.users.exposure_cursor[user] + within, width,
        )
        self.users.exposure_item[user[keep], slot[keep]] = item[keep]
        self.users.exposure_creator[user[keep], slot[keep]] = creator[keep]
        self.users.exposure_topic[user[keep], slot[keep]] = topic[keep]
        self.users.exposure_time[user[keep], slot[keep]] = event_time[keep]
        self.users.exposure_positive[user[keep], slot[keep]] = positive[keep]
        counts = torch.zeros_like(self.users.exposure_cursor)
        counts.index_add_(0, user, torch.ones_like(user))
        self.users.exposure_cursor.add_(counts)

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
