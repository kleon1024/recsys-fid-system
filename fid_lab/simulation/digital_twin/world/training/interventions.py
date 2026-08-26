"""Paired observable interventions backed by the v4 formula stress world."""

from __future__ import annotations

import math

import torch

from ...contracts import EventType
from .....feed_loop.world_model.contracts import STRUCTURAL_INTERVENTION_NAMES
from ..behavior import ResponseTensors, sample_response_tensors
from ..neural_features import build_neural_scm_batch
from ..state import RequestStateOverride


def _selected(values, choice):
    rows = torch.arange(len(choice), device=choice.device)
    return values[rows, choice]


def select_responses(response: ResponseTensors, selector) -> ResponseTensors:
    return ResponseTensors(
        examined=response.examined[selector],
        affinity=response.affinity[selector],
        utility=response.utility[selector],
        dwell_ms=response.dwell_ms[selector],
        action={name: value[selector] for name, value in response.action.items()},
        session_end=(
            None if response.session_end is None
            else response.session_end[selector]
        ),
    )


def _interest_override(snapshot, catalog, slate, choice):
    users = snapshot.users
    user = slate.user_id
    item = _selected(slate.item_ids, choice)
    target = snapshot.catalog_truth.semantic_embedding[item]
    short_interest = torch.nn.functional.normalize(
        0.72 * users.short_interest[user] + 0.28 * target, dim=1,
    )
    long_interest = torch.nn.functional.normalize(
        0.88 * users.long_interest[user] + 0.12 * target, dim=1,
    )
    topic_scale = max(int(catalog.topic_id.max()), 1)
    topic = catalog.topic_id[item].float() / topic_scale
    sequence = users.behavior_sequence[user].float().clone()
    sequence[:, -3:, 0] = topic[:, None]
    sequence[:, -1, 2] = 1.0
    sequence[:, -1, 3] = 1.0
    return RequestStateOverride(
        short_interest=short_interest,
        long_interest=long_interest,
        behavior_sequence=sequence.to(users.behavior_sequence.dtype),
    )


def _quality_override(snapshot, catalog, slate, choice):
    item = slate.item_ids.clamp_min(0)
    selected_item = _selected(item, choice)
    selected = item == selected_item[:, None]
    public_quality = catalog.quality_prior[item]
    treated_public = torch.sigmoid(
        torch.logit(public_quality.clamp(1e-4, 1.0 - 1e-4)) + 0.70
    )
    hidden_quality = snapshot.catalog_truth.quality[item]
    treated_hidden = torch.sigmoid(
        torch.logit(hidden_quality.clamp(1e-4, 1.0 - 1e-4)) + 0.70
    )
    return RequestStateOverride(
        public_quality=torch.where(selected, treated_public, public_quality),
        hidden_quality=torch.where(selected, treated_hidden, hidden_quality),
    )


def _negative_override(snapshot, catalog, slate, choice):
    del catalog
    del choice
    users = snapshot.users
    user = slate.user_id
    sequence = users.behavior_sequence[user].float().clone()
    sequence[:, -1, 5] = 1.0
    sequence[:, -1, 2:5] = 0.0
    return RequestStateOverride(
        behavior_sequence=sequence.to(users.behavior_sequence.dtype),
        fatigue=(users.fatigue[user] + 0.16).clamp(0.0, 1.0),
        satisfaction=(users.satisfaction[user] - 0.10).clamp(0.0, 1.0),
    )


_REQUEST_INTERVENTIONS = (
    _interest_override,
    _quality_override,
    _negative_override,
)


def _observed_value(response, choice):
    stay = _selected(response.dwell_ms, choice).float() / 1_000.0
    action = response.action
    return (
        0.55 * torch.log1p(stay) / math.log(181.0)
        + 0.30 * _selected(action[EventType.LONG_VIEW], choice).float()
        + 0.10 * _selected(action[EventType.LIKE], choice).float()
        - 0.05 * _selected(action[EventType.NEGATIVE], choice).float()
    )


def paired_interventions(snapshot, catalog, slate, control, choice, seed):
    features = []
    slates = []
    sequences = []
    effects = []
    control_value = _observed_value(control, choice)
    rows = torch.arange(len(choice), device=choice.device)
    for treatment in _REQUEST_INTERVENTIONS:
        override = treatment(snapshot, catalog, slate, choice)
        batch = build_neural_scm_batch(
            snapshot, catalog, slate, override,
        )
        response = sample_response_tensors(
            snapshot, catalog, slate, seed, override,
        )
        features.append(batch["slate_features"][rows, choice])
        slates.append(batch["slate_features"])
        sequences.append(batch["sequence"])
        effects.append(_observed_value(response, choice) - control_value)
    if len(features) != len(STRUCTURAL_INTERVENTION_NAMES):
        raise AssertionError("structural intervention contract drift")
    payload = {
        "structural_intervention_features": torch.stack(features, dim=1),
        "structural_intervention_slates": torch.stack(slates, dim=1),
        "structural_intervention_sequences": torch.stack(sequences, dim=1),
        "structural_intervention_effects": torch.stack(effects, dim=1),
    }
    return payload
