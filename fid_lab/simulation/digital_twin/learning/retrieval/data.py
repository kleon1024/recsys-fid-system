"""Load v4 RecallExample rows and public corpus without hidden-world access."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from ...catalog import PublicCatalog
from ...observability import FullFlowPartitionRef, FullFlowSnapshot
from ...platform.projection import ProjectionSnapshot
from ..arrow import list_column_to_tensor
from ..sample_bus import PartitionedSampleBus
from .contracts import RetrievalCorpus, RetrievalQueryBatch


def corpus_from_snapshot(snapshot: FullFlowSnapshot) -> RetrievalCorpus:
    return corpus_from_runtime(
        snapshot.catalog,
        snapshot.projection,
        snapshot.trace.manifest.catalog_version,
    )


def corpus_from_runtime(
    catalog: PublicCatalog,
    projection: ProjectionSnapshot,
    catalog_version: str,
) -> RetrievalCorpus:
    state = projection.state
    return RetrievalCorpus(
        item_id=catalog.item_id.detach().cpu(),
        content_kind=catalog.content_kind.detach().cpu(),
        topic_id=catalog.topic_id.detach().cpu(),
        content_embedding=catalog.content_embedding.detach().cpu(),
        creator_id=state.item_creator_id.detach().cpu(),
        country=state.item_country.detach().cpu(),
        region=state.item_region.detach().cpu(),
        publish_time=state.item_publish_time.detach().cpu(),
        quality_prior=catalog.quality_prior.detach().cpu(),
        duration_seconds=catalog.duration_seconds.detach().cpu(),
        active=state.item_active.detach().cpu(),
        catalog_version=catalog_version,
    )


def _concat(tables: list[pa.Table], name: str) -> pa.Table:
    if not tables:
        raise ValueError(f"retrieval loader requires {name} tables")
    return pa.concat_tables(tables)


def _numpy(table: pa.Table, name: str, dtype) -> np.ndarray:
    return np.array(
        table[name].to_numpy(zero_copy_only=False), dtype=dtype, copy=True,
    )


def load_retrieval_batch(
    bus: PartitionedSampleBus,
    refs: tuple[FullFlowPartitionRef, ...],
) -> RetrievalQueryBatch:
    if not refs:
        raise ValueError("retrieval batch requires at least one partition")
    recall = _concat([bus.recall_examples(ref) for ref in refs], "recall")
    requests = _concat([bus.request_context(ref) for ref in refs], "request")
    positive = recall.filter(pc.equal(recall["role"], "positive"))
    negative = recall.filter(pc.equal(recall["role"], "negative"))
    if not len(positive) or not len(negative):
        raise ValueError("retrieval batch requires positive and negative rows")

    positive_request = _numpy(positive, "request_id", np.int64)
    order = np.argsort(positive_request, kind="stable")
    positive_request = positive_request[order]
    if len(np.unique(positive_request)) != len(positive_request):
        raise ValueError("retrieval positives must contain one row per request")

    request_id = _numpy(requests, "request_id", np.int64)
    request_order = np.argsort(request_id, kind="stable")
    sorted_request = request_id[request_order]
    location = np.searchsorted(sorted_request, positive_request)
    if (
        (location >= len(sorted_request)).any()
        or not np.array_equal(sorted_request[location], positive_request)
    ):
        raise ValueError("retrieval positives lack request context")
    request_rows = request_order[location]

    negative_request = _numpy(negative, "request_id", np.int64)
    negative_ordinal = _numpy(negative, "ordinal", np.int64)
    row = np.searchsorted(positive_request, negative_request)
    if (
        (row >= len(positive_request)).any()
        or not np.array_equal(positive_request[row], negative_request)
    ):
        raise ValueError("retrieval negative lacks a positive request")
    width = int(negative_ordinal.max()) + 1
    negative_item = torch.full((len(positive), width), -1, dtype=torch.long)
    negative_expected = torch.zeros((len(positive), width), dtype=torch.float32)
    negative_mask = torch.zeros((len(positive), width), dtype=torch.bool)
    row_tensor = torch.from_numpy(row)
    ordinal_tensor = torch.from_numpy(negative_ordinal)
    item_tensor = torch.from_numpy(_numpy(negative, "item_id", np.int64))
    expected_tensor = torch.from_numpy(
        _numpy(negative, "sampling_expected_count", np.float32)
    )
    false_negative = torch.from_numpy(
        _numpy(negative, "false_negative_mask", np.bool_)
    )
    negative_item[row_tensor, ordinal_tensor] = item_tensor
    negative_expected[row_tensor, ordinal_tensor] = expected_tensor
    negative_mask[row_tensor, ordinal_tensor] = ~false_negative

    def request_scalar(name: str) -> torch.Tensor:
        return torch.from_numpy(
            _numpy(requests, name, np.int64)[request_rows].copy()
        )

    contract = bus.contract()
    trace = contract["trace_manifest"]
    return RetrievalQueryBatch(
        request_id=torch.from_numpy(positive_request.copy()),
        user_id=request_scalar("user_id"),
        surface=request_scalar("surface"),
        event_time=request_scalar("event_time"),
        query_topic=request_scalar("query_topic"),
        user_country=request_scalar("user_country"),
        user_region=request_scalar("user_region"),
        user_event_counts=list_column_to_tensor(
            requests["user_event_counts"], torch.float32,
        )[request_rows],
        user_surface_counts=list_column_to_tensor(
            requests["user_surface_counts"], torch.float32,
        )[request_rows],
        history_item_id=list_column_to_tensor(
            requests["history_item_id"], torch.long,
        )[request_rows],
        history_event_type=list_column_to_tensor(
            requests["history_event_type"], torch.long,
        )[request_rows],
        positive_item_id=torch.from_numpy(
            _numpy(positive, "item_id", np.int64)[order].copy()
        ),
        positive_strength=torch.from_numpy(
            _numpy(positive, "label_value", np.float32)[order].copy()
        ),
        negative_item_id=negative_item,
        negative_expected_count=negative_expected,
        negative_loss_mask=negative_mask,
        feature_manifest_hash=str(trace["feature_manifest_hash"]),
        partition_content_hashes=tuple(ref.content_sha256 for ref in refs),
        event_watermark=max(ref.event_watermark for ref in refs),
    )
