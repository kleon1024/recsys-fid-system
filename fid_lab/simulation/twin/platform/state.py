"""State available to retrieval, ranking, logging, and training."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


def _clone_dataclass(value):
    return type(value)(**{
        field.name: (
            getattr(value, field.name).clone()
            if isinstance(getattr(value, field.name), torch.Tensor)
            else getattr(value, field.name)
        )
        for field in fields(value)
    })


@dataclass
class ExposureLedger:
    item: torch.Tensor
    author: torch.Tensor
    cluster: torch.Tensor
    topic: torch.Tensor
    kind: torch.Tensor
    surface: torch.Tensor
    step: torch.Tensor

    def clone(self):
        return _clone_dataclass(self)

@dataclass
class UserState:
    user_id: torch.Tensor
    short_interest: torch.Tensor
    observed_interest: torch.Tensor
    surface_affinity_estimate: torch.Tensor
    commerce_intent_estimate: torch.Tensor
    local_intent_estimate: torch.Tensor
    creator_intent_estimate: torch.Tensor
    query_topic: torch.Tensor
    query_strength: torch.Tensor
    satisfaction_estimate: torch.Tensor
    fatigue_counter: torch.Tensor
    lifecycle: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    timezone_offset: torch.Tensor
    socioeconomic: torch.Tensor
    spending_power_estimate: torch.Tensor
    activity_tier: torch.Tensor
    activity_rate_estimate: torch.Tensor
    acquisition_channel: torch.Tensor
    signup_step: torch.Tensor
    tenure_days: torch.Tensor
    cold_start_confidence: torch.Tensor
    trend_affinity_estimate: torch.Tensor
    registered: torch.Tensor
    active: torch.Tensor
    session_depth: torch.Tensor
    request_index: torch.Tensor
    ledger: ExposureLedger

    def clone(self):
        values = {
            field.name: (
                self.ledger.clone()
                if field.name == "ledger"
                else getattr(self, field.name).clone()
            )
            for field in fields(self)
        }
        return UserState(**values)


def select_users(users: UserState, selector) -> UserState:
    values = {}
    for field in fields(users):
        value = getattr(users, field.name)
        if field.name == "ledger":
            values[field.name] = ExposureLedger(**{
                ledger_field.name: getattr(value, ledger_field.name)[
                    selector
                ].clone()
                for ledger_field in fields(value)
            })
        else:
            values[field.name] = value[selector].clone()
    return UserState(**values)


def writeback_users(users: UserState, selected: UserState, selector) -> None:
    for field in fields(users):
        target = getattr(users, field.name)
        source = getattr(selected, field.name)
        if field.name == "ledger":
            for ledger_field in fields(target):
                getattr(target, ledger_field.name)[selector] = getattr(
                    source, ledger_field.name
                )
        else:
            target[selector] = source


@dataclass
class CatalogState:
    item_id: torch.Tensor
    kind: torch.Tensor
    topic: torch.Tensor
    topic_embedding: torch.Tensor
    author: torch.Tensor
    cluster: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    quality: torch.Tensor
    text_quality: torch.Tensor
    visual_quality: torch.Tensor
    duration_seconds: torch.Tensor
    freshness: torch.Tensor
    popularity: torch.Tensor
    risk: torch.Tensor
    price_match_prior: torch.Tensor
    price: torch.Tensor
    merchant_quality: torch.Tensor
    inventory: torch.Tensor
    sponsored_value: torch.Tensor
    ad_bid: torch.Tensor
    ad_budget: torch.Tensor
    ad_spend: torch.Tensor
    live_start_hour: torch.Tensor
    live_duration_hours: torch.Tensor
    poi_open_hour: torch.Tensor
    poi_close_hour: torch.Tensor
    supply_exposure: torch.Tensor
    supply_positive: torch.Tensor
    supply_negative: torch.Tensor
    supply_payment: torch.Tensor
    creator_motivation: torch.Tensor
    creator_active: torch.Tensor
    creator_posts: torch.Tensor

    def clone(self):
        return _clone_dataclass(self)
