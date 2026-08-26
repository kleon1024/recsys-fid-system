"""Delayed cross-request labels for the Feed Publish Queue."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import torch

from ..contracts import AppEventBatch, EventType, Surface
from .contracts import FineRankExampleBatch, PublishQueueExampleBatch
from .event_closure import PUBLISH_QUEUE_SOURCE_TYPES


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
    attributed to one globally observable, engaged last-touch Feed exposure.
    This prevents one outcome from being credited once per request partition;
    causal lift still requires randomized serving and A/B evaluation.
    """

    def __init__(self, config: PublishQueueConfig):
        self.config = config

    @staticmethod
    def _source_index(events: AppEventBatch):
        source_strength = {
            event_type: strength
            for strength, event_type in enumerate(PUBLISH_QUEUE_SOURCE_TYPES)
        }
        source = (
            (events.surface == int(Surface.FEED))
            & (events.user_id >= 0)
            & (events.item_id >= 0)
            & torch.isin(events.event_type, torch.tensor(
                [int(event_type) for event_type in PUBLISH_QUEUE_SOURCE_TYPES],
                device=events.event_type.device,
            ))
        )
        by_user: dict[int, list[tuple[int, int, int, int, int]]] = {}
        for row in torch.where(source)[0].detach().cpu().tolist():
            event_type = EventType(int(events.event_type[row]))
            by_user.setdefault(int(events.user_id[row]), []).append((
                int(events.event_time[row]),
                source_strength[event_type],
                int(events.event_id[row]),
                int(events.request_id[row]),
                int(events.item_id[row]),
            ))
        times = {}
        for outcome_user, values in by_user.items():
            values.sort()
            times[outcome_user] = [value[0] for value in values]
        return by_user, times

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

        exposure_rows, exposure_columns = torch.where(exposed)
        exposure_index: dict[tuple[int, int, int], tuple[int, int]] = {}
        for row, column in zip(
            exposure_rows.detach().cpu().tolist(),
            exposure_columns.detach().cpu().tolist(),
        ):
            exposure_index[(
                int(user_id[row]),
                int(fine.request_id[selected][row]),
                int(item[row, column]),
            )] = (row, column)

        source_by_user, source_times = self._source_index(events)

        for task_index, task in enumerate(tasks):
            outcome = (
                events.event(task.event_type)
                & (events.user_id >= 0)
                & (events.surface == int(Surface.POSTING))
                & (events.event_time <= event_watermark)
            )
            outcome_users = events.user_id[outcome].detach().cpu().tolist()
            outcome_times = events.event_time[outcome].detach().cpu().tolist()
            outcome_ids = events.event_id[outcome].detach().cpu().tolist()
            for outcome_user, outcome_time, outcome_id in zip(
                outcome_users, outcome_times, outcome_ids, strict=True,
            ):
                candidates = source_by_user.get(int(outcome_user))
                if not candidates:
                    continue
                times = source_times[int(outcome_user)]
                start = bisect_left(times, int(outcome_time) - task.window_ticks)
                stop = bisect_right(times, int(outcome_time))
                if start == stop:
                    continue
                _, _, _, source_request, source_item = max(
                    candidates[start:stop]
                )
                location = exposure_index.get((
                    int(outcome_user), source_request, source_item,
                ))
                if location is not None:
                    row, column = location
                    labels[row, column, task_index] = 1.0
                    attribution_event_id[row, column, task_index] = outcome_id

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
            attribution_method="engaged_last_touch_v1",
            feature_manifest_hash=fine.feature_manifest_hash,
        )
