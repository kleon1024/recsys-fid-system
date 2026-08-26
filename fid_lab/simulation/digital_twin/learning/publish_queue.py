"""Persisted Publish Queue examples and serving-value ownership."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from ..observability import FullFlowPartitionRef
from .arrow import list_column_to_tensor
from .contracts import ProbeBatch
from .sample_bus import PartitionedSampleBus


PUBLISH_QUEUE_VALUE_VERSION = "feed-publish-queue-v1"
PUBLISH_QUEUE_TASK_WEIGHTS = {
    "posting_entry_24h": 0.10,
    "create_24h": 0.30,
    "publish_48h": 0.60,
}


def publish_queue_task_weights(task_names: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(PUBLISH_QUEUE_TASK_WEIGHTS.get(name, 0.0) for name in task_names)


def load_publish_queue_batch(
    bus: PartitionedSampleBus,
    refs: tuple[FullFlowPartitionRef, ...],
) -> ProbeBatch:
    if not refs:
        raise ValueError("Publish Queue requires at least one partition")
    tables = [bus.publish_queue_examples(ref) for ref in refs]
    table = pa.concat_tables(tables)
    table = table.filter(pc.equal(table["exposed"], True))
    if not len(table):
        raise ValueError("partitions contain no factual Feed exposures")
    contract = bus.contract()
    sample = contract["sample_contract"]
    trace = contract["trace_manifest"]
    order = np.lexsort((
        table["ordinal"].to_numpy(),
        table["request_id"].to_numpy(),
        table["request_time"].to_numpy(),
    ))

    def scalar(name: str, dtype: torch.dtype) -> torch.Tensor:
        values = table[name].to_numpy(zero_copy_only=False)[order]
        return torch.as_tensor(values.copy(), dtype=dtype)

    def listed(name: str, dtype: torch.dtype) -> torch.Tensor:
        return list_column_to_tensor(table[name], dtype)[order]

    labels = listed("task_label_values", torch.float32)
    masks = listed("task_label_masks", torch.bool)
    applicable = listed("task_label_applicable", torch.bool)
    mature = listed("task_label_mature", torch.bool)
    rows = len(table)
    return ProbeBatch(
        request_id=scalar("request_id", torch.long),
        user_id=scalar("user_id", torch.long),
        surface=scalar("surface", torch.long),
        request_time=scalar("request_time", torch.long),
        item_id=scalar("item_id", torch.long),
        position=scalar("position", torch.long),
        route_id=torch.full((rows,), -1, dtype=torch.long),
        recall_score=torch.zeros(rows),
        exposed=torch.ones(rows, dtype=torch.bool),
        candidate_exposure_probability=scalar(
            "candidate_exposure_probability", torch.float32,
        ),
        randomized_support=torch.zeros(rows, dtype=torch.bool),
        dwell_ms=torch.zeros(rows),
        dense_features=listed("dense_features", torch.float32),
        sparse_buckets=listed("sparse_buckets", torch.long),
        labels=labels,
        label_mask=masks,
        label_applicable=applicable,
        label_mature=mature,
        joint_logging_probability=scalar(
            "joint_logging_probability", torch.float32,
        ),
        task_names=tuple(sample["publish_queue_task_names"]),
        dense_feature_names=tuple(sample["dense_feature_names"]),
        sparse_feature_names=tuple(sample["sparse_feature_names"]),
        feature_manifest_hash=str(trace["feature_manifest_hash"]),
        partition_content_hashes=tuple(ref.content_sha256 for ref in refs),
        event_watermark=max(ref.event_watermark for ref in refs),
    )
