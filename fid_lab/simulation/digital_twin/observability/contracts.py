"""Typed inputs for the durable v4 full-flow analytical authority."""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import PublicCatalog
from ..contracts import AppEventBatch
from ..engine import LayerAssignmentTrace
from ..platform.projection import ProjectionSnapshot
from ..samples.contracts import (
    JoinedSampleAuthorities,
    RequestCandidateTrace,
    RequestContextBatch,
)


@dataclass(frozen=True)
class CheckpointRecord:
    created_time: int
    lane: str
    model_name: str
    checkpoint_version: str
    data_watermark: int
    sample_manifest: str
    feature_version: str
    fid_version: str
    index_version: str
    validation_status: str
    publish_state: str
    fallback_version: str = ""


@dataclass(frozen=True)
class FullFlowSnapshot:
    catalog: PublicCatalog
    trace: RequestCandidateTrace
    context: RequestContextBatch
    events: AppEventBatch
    samples: JoinedSampleAuthorities
    projection: ProjectionSnapshot
    checkpoints: tuple[CheckpointRecord, ...] = ()
    layer_assignment: LayerAssignmentTrace | None = None

    def __post_init__(self):
        if not self.trace.request_id.equal(self.context.request_id):
            raise ValueError("full-flow trace and context are misaligned")
        if self.samples.manifest != self.trace.manifest:
            raise ValueError("full-flow sample and trace manifests differ")
        if self.projection.as_of_ingest_time < int(self.trace.event_time.max()):
            raise ValueError("full-flow projection predates request trace")
        if self.layer_assignment is not None and not self.trace.request_id.equal(
            self.layer_assignment.request_id
        ):
            raise ValueError("full-flow layer assignments are misaligned")
