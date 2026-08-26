"""Materialize the factual request evidence behind one Launch Review."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

from ...engine import TickResult
from ...observability import FullFlowSnapshot, append_full_flow_partition
from ...observability.launch_diagnose import write_diagnosis
from ...observability.store import replace_json_atomic
from ...samples.contracts import RequestCandidateTrace, RequestContextBatch
from ...samples.event_closure import select_joiner_events
from ...samples.joiner import JoinerConfig, RequestLevelJoiner
from ...samples.publish_queue import PublishQueueConfig


@dataclass(frozen=True)
class EvidenceTick:
    logical_time: int
    candidate_trace: RequestCandidateTrace
    request_context: RequestContextBatch


def _move_tensor_fields(value, device: torch.device):
    return type(value)(**{
        item.name: (
            getattr(value, item.name).detach().to(device)
            if isinstance(getattr(value, item.name), torch.Tensor)
            else getattr(value, item.name)
        )
        for item in fields(value)
    })


@dataclass
class LaunchEvidenceCollector:
    """Bounded lifetime collector; durable ownership moves to Parquet."""

    ticks: list[EvidenceTick] = field(default_factory=list)

    def append(self, tick: TickResult) -> None:
        if tick.candidate_trace is not None:
            self.ticks.append(
                EvidenceTick(
                    tick.logical_time,
                    _move_tensor_fields(
                        tick.candidate_trace, torch.device("cpu"),
                    ),
                    _move_tensor_fields(
                        tick.request_context, torch.device("cpu"),
                    ),
                )
            )

    def materialize(
        self,
        *,
        kernel,
        output_dir: Path,
        review: dict[str, object],
        ticks_per_day: int,
    ) -> dict[str, object]:
        if not self.ticks:
            raise ValueError("launch review emitted no candidate evidence")
        output_dir.mkdir(parents=True, exist_ok=True)
        replace_json_atomic(output_dir / "ab-result.json", review)
        all_events = kernel.event_log.read(
            ingested_through=int(review["analysis_end_time"]),
        )
        joiner = RequestLevelJoiner(
            JoinerConfig(ticks_per_day=ticks_per_day),
            kernel.platform.catalog,
        )
        publish_window_ticks = max(
            task.window_ticks
            for task in PublishQueueConfig(ticks_per_day).tasks
        )
        full_flow = output_dir / "full-flow-dataset"
        projection = kernel.platform.projection.snapshot()
        feature_manifest = kernel.platform.ranker.features.manifest
        requests = 0
        event_count = 0
        manifest = None
        while self.ticks:
            tick = self.ticks.pop(0)
            device = kernel.platform.catalog.item_id.device
            trace = _move_tensor_fields(tick.candidate_trace, device)
            context = _move_tensor_fields(tick.request_context, device)
            join_events = select_joiner_events(
                all_events,
                request_id=trace.request_id,
                user_id=trace.user_id,
                request_time=trace.event_time,
                publish_window_ticks=publish_window_ticks,
            )
            samples = joiner.materialize(
                trace,
                context,
                join_events,
                event_watermark=int(review["analysis_end_time"]),
            )
            persist = torch.isin(all_events.request_id, trace.request_id)
            events = all_events.select(persist)
            snapshot = FullFlowSnapshot(
                catalog=kernel.platform.catalog,
                trace=trace,
                context=context,
                events=events,
                samples=samples,
                projection=projection,
                feature_manifest=feature_manifest,
            )
            manifest = append_full_flow_partition(
                snapshot, full_flow, f"event_time={tick.logical_time}",
            )
            requests += len(trace.request_id)
            event_count += len(events.event_id)
            del samples, snapshot, events, join_events, trace, context, tick
        diagnosis = write_diagnosis(full_flow, output_dir, review=review)
        return {
            "path": str(output_dir),
            "dataset_content_sha256": manifest["dataset_content_sha256"],
            "requests": requests,
            "events": event_count,
            "findings": diagnosis["findings"],
        }
