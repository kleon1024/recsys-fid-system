"""Delayed cross-request labels for the Feed Publish Queue."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

import torch

from ..contracts import AppEventBatch, EventType, Surface
from .contracts import FineRankExampleBatch, PublishQueueExampleBatch


@dataclass(frozen=True)
class PublishQueueTask:
    name: str
    event_type: EventType
    window_ticks: int


@dataclass(frozen=True)
class PublishQueueConfig:
    ticks_per_day: int

    def __post_init__(self):
        if self.ticks_per_day <= 0:
            raise ValueError("publish-queue dimensions must be positive")

    @property
    def tasks(self) -> tuple[PublishQueueTask, ...]:
        day = self.ticks_per_day
        return (
            PublishQueueTask("posting_entry_24h", EventType.SURFACE_ENTRY, day),
            PublishQueueTask("create_24h", EventType.CREATE, day),
            PublishQueueTask("publish_48h", EventType.PUBLISH, 2 * day),
        )


class PublishQueueJoiner:
    """Attribute later creator outcomes to observable Feed exposures.

    Hidden inspiration state is deliberately unavailable here.  Outcomes are
    Each factual Feed exposure receives a delayed binary outcome. Multiple
    exposures may share one later publication because the target is conditional
    response probability, not a claim of unique causal attribution. Causal lift
    still requires randomized serving and A/B evaluation.
    """

    def __init__(self, config: PublishQueueConfig):
        self.config = config

    def materialize(
        self,
        fine: FineRankExampleBatch,
        events: AppEventBatch,
        event_watermark: int,
    ) -> PublishQueueExampleBatch:
        selected = fine.surface == int(Surface.FEED)
        context = fine.context.select(selected)
        item = fine.item_id[selected]
        exposed = fine.exposed[selected] & (item >= 0)
        request_time = fine.request_time[selected]
        user_id = fine.user_id[selected]
        tasks = self.config.tasks
        task_count = len(tasks)
        shape = (*item.shape, task_count)
        labels = torch.zeros(shape, dtype=torch.float, device=item.device)
        attribution_event_id = torch.full(
            shape, -1, dtype=torch.long, device=item.device,
        )

        maturity_time = torch.stack(tuple(
            request_time[:, None].expand_as(item) + task.window_ticks
            for task in tasks
        ), dim=2)
        mature = (
            exposed[:, :, None]
            & (event_watermark >= maturity_time)
        )

        for task_index, task in enumerate(tasks):
            outcome = (
                events.event(task.event_type)
                & (events.user_id >= 0)
                & (events.surface == int(Surface.POSTING))
                & (events.event_time <= event_watermark)
            )
            by_user: dict[int, list[tuple[int, int]]] = {}
            for outcome_time, outcome_user, outcome_id in zip(
                events.event_time[outcome].detach().cpu().tolist(),
                events.user_id[outcome].detach().cpu().tolist(),
                events.event_id[outcome].detach().cpu().tolist(),
                strict=True,
            ):
                by_user.setdefault(int(outcome_user), []).append((
                    int(outcome_time), int(outcome_id),
                ))
            for values in by_user.values():
                values.sort()
            for row, (exposure_user, exposure_time) in enumerate(zip(
                user_id.detach().cpu().tolist(),
                request_time.detach().cpu().tolist(),
                strict=True,
            )):
                outcomes = by_user.get(int(exposure_user), ())
                location = bisect_left(outcomes, (int(exposure_time), -1))
                if location == len(outcomes):
                    continue
                outcome_time, outcome_id = outcomes[location]
                if outcome_time > int(exposure_time) + task.window_ticks:
                    continue
                labels[row, exposed[row], task_index] = 1.0
                attribution_event_id[row, exposed[row], task_index] = outcome_id

        labels = labels.clamp_max(1.0) * mature.float()
        attribution_event_id = torch.where(
            mature, attribution_event_id, torch.full_like(attribution_event_id, -1),
        )
        exposure_probability = fine.exposure_probability[selected]
        assignment_probability = fine.assignment_probability[selected]
        return PublishQueueExampleBatch(
            request_id=fine.request_id[selected],
            user_id=user_id,
            request_time=request_time,
            item_id=item,
            position=fine.position[selected],
            exposed=exposed,
            exposure_probability=exposure_probability,
            assignment_probability=assignment_probability,
            joint_logging_probability=(
                exposure_probability * assignment_probability[:, None]
            ),
            labels=labels,
            label_mature=mature,
            label_mask=mature,
            label_maturity_time=maturity_time,
            attribution_event_id=attribution_event_id,
            dense_features=fine.dense_features[selected],
            sparse_fids=fine.sparse_fids[selected],
            sparse_buckets=fine.sparse_buckets[selected],
            context=context,
            task_names=tuple(task.name for task in tasks),
            task_window_ticks=tuple(task.window_ticks for task in tasks),
            attribution_method="exposure_window_outcome_v1",
            feature_manifest_hash=fine.feature_manifest_hash,
        )
