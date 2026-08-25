"""Hidden user-memory effects that the recommendation platform cannot observe."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..catalog import PublicCatalog
from ..contracts import ContentKind, RenderedSlateBatch, Surface
from .state import HiddenUserState, UserWorldSnapshot


EXPERIENCE_DYNAMICS_VERSION = "hidden-experience-v1"


@dataclass(frozen=True)
class CandidateExperience:
    exact_repeat: torch.Tensor
    creator_pressure: torch.Tensor
    topic_pressure: torch.Tensor
    rewarded_rewatch: torch.Tensor
    repeat_penalty: torch.Tensor

    @property
    def slate_repeat_pressure(self) -> torch.Tensor:
        return self.repeat_penalty.mean(dim=1)


def candidate_experience(
    snapshot: UserWorldSnapshot,
    catalog: PublicCatalog,
    slate: RenderedSlateBatch,
) -> CandidateExperience:
    item = slate.item_ids.clamp_min(0)
    feed = (slate.surface == int(Surface.FEED))[:, None]
    media = (
        (catalog.content_kind[item] == int(ContentKind.SHORT_VIDEO))
        | (catalog.content_kind[item] == int(ContentKind.PHOTO))
        | (catalog.content_kind[item] == int(ContentKind.ARTICLE))
        | (catalog.content_kind[item] == int(ContentKind.CARD))
    )
    return experience_for_candidates(
        snapshot.users,
        item=item,
        creator=catalog.creator_id[item],
        topic=catalog.topic_id[item],
        event_time=slate.event_time,
        ticks_per_day=snapshot.ticks_per_day,
        user_id=slate.user_id,
        strict_repeat=feed & media,
    )


def experience_for_candidates(
    users: HiddenUserState,
    *,
    item: torch.Tensor,
    creator: torch.Tensor,
    topic: torch.Tensor,
    event_time: torch.Tensor,
    ticks_per_day: int,
    user_id: torch.Tensor | None = None,
    strict_repeat: torch.Tensor | None = None,
) -> CandidateExperience:
    """Calculate decayed repeat pressure for one or more candidates per user."""
    if item.ndim == 1:
        item = item[:, None]
        creator = creator[:, None]
        topic = topic[:, None]
    row = users.user_id if user_id is None else user_id
    history_item = users.exposure_item[row]
    history_creator = users.exposure_creator[row]
    history_topic = users.exposure_topic[row]
    history_time = users.exposure_time[row]
    history_positive = users.exposure_positive[row].float()
    age = event_time[:, None] - history_time
    valid = (history_item >= 0) & (age >= 0)
    decay = torch.exp(
        -age.clamp_min(0).float() / max(3.0 * ticks_per_day, 1.0),
    ) * valid.float()
    exact_match = item[:, :, None] == history_item[:, None, :]
    creator_match = creator[:, :, None] == history_creator[:, None, :]
    topic_match = topic[:, :, None] == history_topic[:, None, :]
    exact_repeat = (exact_match * decay[:, None]).sum(dim=2)
    rewarded_rewatch = (
        exact_match * decay[:, None] * history_positive[:, None]
    ).sum(dim=2).clamp_max(1.0)
    creator_pressure = (creator_match * decay[:, None]).sum(dim=2)
    topic_pressure = (topic_match * decay[:, None]).sum(dim=2)
    sensitivity = (
        0.42
        + 0.82 * users.novelty[row]
        + 0.12 * torch.sigmoid(users.response_style[row, 6])
    )[:, None]
    exact_unrewarded = (exact_repeat - 0.68 * rewarded_rewatch).clamp_min(0.0)
    repeat_penalty = (
        sensitivity * torch.log1p(exact_unrewarded)
        + 0.10 * torch.log1p(creator_pressure)
        + 0.045 * torch.log1p(topic_pressure)
        - 0.16 * users.habit[row, None] * rewarded_rewatch
    ).clamp_min(0.0)
    if strict_repeat is not None:
        strict_penalty = 3.2 + 0.45 * torch.log1p(exact_repeat)
        repeat_penalty = torch.where(
            strict_repeat & (exact_repeat > 0),
            torch.maximum(repeat_penalty, strict_penalty),
            repeat_penalty,
        )
    return CandidateExperience(
        exact_repeat,
        creator_pressure,
        topic_pressure,
        rewarded_rewatch,
        repeat_penalty,
    )
