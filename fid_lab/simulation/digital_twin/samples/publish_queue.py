"""Delayed cross-request labels for the Feed Publish Queue."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import exp, log

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
    attribution_half_life_ticks: int | None = None
    maximum_attributed_exposures: int = 32

    def __post_init__(self):
        if self.ticks_per_day <= 0 or self.maximum_attributed_exposures <= 0:
            raise ValueError("publish-queue dimensions must be positive")
        if (
            self.attribution_half_life_ticks is not None
            and self.attribution_half_life_ticks <= 0
        ):
            raise ValueError("publish-queue attribution half-life must be positive")

    @property
    def half_life_ticks(self) -> int:
        return self.attribution_half_life_ticks or max(1, self.ticks_per_day // 2)

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
    attributed across recent factual Feed exposures with normalized temporal
    decay.  The label is therefore an observational fractional-credit target;
    causal lift still requires randomized serving and A/B evaluation.
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

        maturity_time = torch.stack(tuple(
            request_time[:, None].expand_as(item) + task.window_ticks
            for task in tasks
        ), dim=2)
        mature = (
            exposed[:, :, None]
            & (event_watermark >= maturity_time)
        )

        exposure_rows, exposure_columns = torch.where(exposed)
        exposure_index: dict[int, list[tuple[int, int, int]]] = {}
        for row, column in zip(
            exposure_rows.detach().cpu().tolist(),
            exposure_columns.detach().cpu().tolist(),
        ):
            exposure_index.setdefault(int(user_id[row]), []).append((
                int(request_time[row]), row, column,
            ))
        for values in exposure_index.values():
            values.sort(key=lambda value: value[0])
        exposure_times = {
            user: [value[0] for value in values]
            for user, values in exposure_index.items()
        }

        for task_index, task in enumerate(tasks):
            outcome = (
                events.event(task.event_type)
                & (events.user_id >= 0)
                & (events.surface == int(Surface.POSTING))
                & (events.event_time <= event_watermark)
            )
            outcome_users = events.user_id[outcome].detach().cpu().tolist()
            outcome_times = events.event_time[outcome].detach().cpu().tolist()
            for outcome_user, outcome_time in zip(outcome_users, outcome_times):
                candidates = exposure_index.get(int(outcome_user))
                if not candidates:
                    continue
                times = exposure_times[int(outcome_user)]
                start = bisect_left(times, int(outcome_time) - task.window_ticks)
                stop = bisect_right(times, int(outcome_time))
                touches = candidates[start:stop]
                if not touches:
                    continue
                touches = touches[-self.config.maximum_attributed_exposures:]
                weights = [
                    exp(
                        -log(2.0)
                        * max(0, int(outcome_time) - exposure_time)
                        / self.config.half_life_ticks
                    )
                    for exposure_time, _, _ in touches
                ]
                denominator = sum(weights)
                for (_, row, column), weight in zip(touches, weights):
                    labels[row, column, task_index] += weight / denominator

        labels = labels.clamp_max(1.0) * mature.float()
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
            dense_features=fine.dense_features[selected],
            sparse_fids=fine.sparse_fids[selected],
            sparse_buckets=fine.sparse_buckets[selected],
            context=context,
            task_names=tuple(task.name for task in tasks),
            task_window_ticks=tuple(task.window_ticks for task in tasks),
            attribution_half_life_ticks=self.config.half_life_ticks,
            feature_manifest_hash=fine.feature_manifest_hash,
        )
