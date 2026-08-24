"""One online/offline authority for ranker-visible behavior sequences."""

from __future__ import annotations

import torch


SEQUENCE_FIELDS = (
    "topic_norm", "stay_norm", "long_view", "quality_long_view", "like",
    "negative_feedback", "anchor_click", "conversion",
)


def initialize_ranking_sequence(state, topic_lookup):
    items = state["behavior_history_items"]
    feedback = state["behavior_history_feedback"].float()
    topic = topic_lookup[items].float() / 11.0
    zeros = torch.zeros_like(topic)
    state["ranking_behavior_sequence"] = torch.stack((
        topic,
        feedback[:, :, 1],
        feedback[:, :, 1],
        feedback[:, :, 1],
        feedback[:, :, 2],
        feedback[:, :, 6],
        zeros,
        zeros,
    ), dim=2)


def append_ranking_event(state, selected, values):
    if "ranking_behavior_sequence" not in state:
        return
    conversion = values["paid"] | values["pixel"]
    event = torch.stack((
        selected["candidate_topic"].float() / 11.0,
        torch.log1p(values["stay"]) / torch.log(
            torch.tensor(181.0, device=values["stay"].device)
        ),
        values["long_view"].float(),
        values["quality_view"].float(),
        values["like"].float(),
        values["negative"].float(),
        values["anchor"].float(),
        conversion.float(),
    ), dim=1)
    sequence = torch.roll(state["ranking_behavior_sequence"], shifts=-1, dims=1)
    sequence[:, -1] = event
    state["ranking_behavior_sequence"] = sequence
