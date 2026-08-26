"""Materialize the factual request evidence behind one Launch Review."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ...engine import TickResult
from ...observability import FullFlowSnapshot, append_full_flow_partition
from ...observability.launch_diagnose import write_diagnosis
from ...samples.joiner import JoinerConfig, RequestLevelJoiner


@dataclass
class LaunchEvidenceCollector:
    """Bounded lifetime collector; durable ownership moves to Parquet."""

    ticks: list[TickResult] = field(default_factory=list)

    def append(self, tick: TickResult) -> None:
        if tick.candidate_trace is not None:
            self.ticks.append(tick)

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
        all_events = kernel.event_log.read(
            ingested_through=int(review["analysis_end_time"]),
        )
        joiner = RequestLevelJoiner(
            JoinerConfig(ticks_per_day=ticks_per_day),
            kernel.platform.catalog,
        )
        full_flow = output_dir / "full-flow-dataset"
        requests = 0
        event_count = 0
        manifest = None
        for tick in self.ticks:
            trace = tick.candidate_trace
            context = tick.request_context
            events = all_events.select(torch.isin(
                all_events.request_id, trace.request_id,
            ))
            samples = joiner.materialize(
                trace,
                context,
                events,
                event_watermark=int(review["analysis_end_time"]),
            )
            snapshot = FullFlowSnapshot(
                catalog=kernel.platform.catalog,
                trace=trace,
                context=context,
                events=events,
                samples=samples,
                projection=kernel.platform.projection.snapshot(),
                feature_manifest=kernel.platform.ranker.features.manifest,
            )
            manifest = append_full_flow_partition(
                snapshot, full_flow, f"event_time={tick.logical_time}",
            )
            requests += len(trace.request_id)
            event_count += len(events.event_id)
        diagnosis = write_diagnosis(full_flow, output_dir, review=review)
        return {
            "path": str(output_dir),
            "dataset_content_sha256": manifest["dataset_content_sha256"],
            "requests": requests,
            "events": event_count,
            "findings": diagnosis["findings"],
        }
