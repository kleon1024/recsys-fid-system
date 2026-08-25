"""Two independent cursors over one immutable factual partition stream."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from filelock import FileLock
import pyarrow.compute as pc

from ..observability import (
    FullFlowPartitionRef,
    list_full_flow_partitions,
    read_full_flow_partition_table,
    replace_json_atomic,
    verify_full_flow_dataset,
)
from .contracts import (
    ArtifactCompatibility,
    Lane,
    LaneCursor,
    content_hash,
    learning_source_hash,
)


class PartitionedSampleBus:
    def __init__(self, dataset_root: Path, state_root: Path) -> None:
        self.dataset_root = dataset_root
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)

    def contract(self) -> dict[str, object]:
        return verify_full_flow_dataset(self.dataset_root)["contract"]

    @property
    def contract_hash(self) -> str:
        return content_hash(self.contract())

    def _cursor_path(self, lane: Lane) -> Path:
        return self.state_root / f"{lane.value}-cursor.json"

    def cursor(self, lane: Lane) -> LaneCursor:
        path = self._cursor_path(lane)
        if not path.exists():
            return LaneCursor(lane, self.contract_hash)
        import json

        cursor = LaneCursor.from_manifest(json.loads(path.read_text()))
        if cursor.lane is not lane:
            raise ValueError("lane cursor identity differs from its file")
        if cursor.contract_hash != self.contract_hash:
            raise ValueError("lane cursor dataset contract changed")
        return cursor

    def poll(
        self,
        lane: Lane,
        *,
        limit: int | None = None,
        max_watermark: int | None = None,
    ) -> tuple[FullFlowPartitionRef, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("poll limit must be positive")
        cursor = self.cursor(lane)
        consumed = dict(cursor.consumed)
        available = tuple(
            ref for ref in list_full_flow_partitions(self.dataset_root)
            if ref.key not in consumed
            and (max_watermark is None or ref.event_watermark <= max_watermark)
        )
        return available if limit is None else available[:limit]

    def commit(self, lane: Lane, ref: FullFlowPartitionRef) -> dict[str, object]:
        path = self._cursor_path(lane)
        with FileLock(str(path.with_suffix(".lock"))):
            cursor = self.cursor(lane)
            consumed = dict(cursor.consumed)
            if ref.key in consumed:
                if consumed[ref.key] != ref.content_sha256:
                    raise ValueError("consumed partition content changed")
                return {"status": "resumed", **cursor.manifest()}
            pending = self.poll(lane, limit=1)
            if not pending or pending[0] != ref:
                raise ValueError("lane commits must follow event-time order")
            updated = replace(
                cursor,
                consumed=(*cursor.consumed, (ref.key, ref.content_sha256)),
                event_watermark=ref.event_watermark,
            )
            replace_json_atomic(path, updated.manifest())
            return {"status": "committed", **updated.manifest()}

    def fine_examples(self, ref: FullFlowPartitionRef):
        table = read_full_flow_partition_table(
            self.dataset_root, ref, "v4_training_example_log",
        )
        return table.filter(pc.equal(table["authority"], "fine"))

    def compatibility(
        self,
        *,
        index_version: str,
        corpus_sha256: str,
    ) -> ArtifactCompatibility:
        contract = self.contract()
        trace = contract["trace_manifest"]
        return ArtifactCompatibility(
            dataset_contract_hash=self.contract_hash,
            feature_manifest_hash=str(trace["feature_manifest_hash"]),
            feature_version=str(trace["feature_version"]),
            fid_version=str(trace["fid_version"]),
            catalog_version=str(trace["catalog_version"]),
            index_version=index_version,
            corpus_sha256=corpus_sha256,
            code_sha256=learning_source_hash(),
        )
