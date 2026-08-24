"""Vectorized Arrow tables derived from one factual v4 snapshot."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pyarrow as pa
import torch

from ..contracts import EventType
from .contracts import CheckpointRecord, FullFlowSnapshot


TABLE_NAMES = (
    "v4_request_log",
    "v4_route_candidate_log",
    "v4_candidate_decision_log",
    "v4_event_log",
    "v4_mature_label_log",
    "v4_training_example_log",
    "v4_checkpoint_log",
)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _fixed_list(value: torch.Tensor) -> pa.FixedSizeListArray:
    if value.ndim != 2:
        raise ValueError("fixed-list source must be two-dimensional")
    flat = pa.array(_numpy(value).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, value.shape[1])


def _stage_membership(
    parent: torch.Tensor,
    child: torch.Tensor,
    child_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match = (
        (parent[:, :, None] >= 0)
        & (child[:, None, :] >= 0)
        & (parent[:, :, None] == child[:, None, :])
    )
    present = match.any(dim=2)
    location = match.float().argmax(dim=2)
    score = torch.gather(child_score, 1, location)
    score = torch.where(present, score, torch.full_like(score, torch.nan))
    rank = torch.where(
        present,
        location + 1,
        torch.full_like(location, -1),
    )
    return present, rank, score


def _request_table(snapshot: FullFlowSnapshot) -> pa.Table:
    trace, context = snapshot.trace, snapshot.context
    data: dict[str, object] = {
        "request_id": _numpy(trace.request_id),
        "user_id": _numpy(trace.user_id),
        "surface": _numpy(trace.surface),
        "event_time": _numpy(trace.event_time),
        "query_topic": _numpy(trace.query_topic),
        "user_country": _numpy(trace.user_country),
        "user_region": _numpy(trace.user_region),
        "experiment_cell": _numpy(trace.experiment_cell),
        "assignment_probability": _numpy(trace.assignment_probability),
        "recall_version_id": _numpy(trace.recall_version_id),
        "coarse_version_id": _numpy(trace.coarse_version_id),
        "fine_version_id": _numpy(trace.fine_version_id),
        "mix_version_id": _numpy(trace.mix_version_id),
        "feature_as_of_ingest_time": _numpy(
            context.feature_as_of_ingest_time
        ),
        "trace_schema_version": [trace.manifest.schema_version] * len(trace.request_id),
        "feature_version": [trace.manifest.feature_version] * len(trace.request_id),
        "catalog_version": [trace.manifest.catalog_version] * len(trace.request_id),
        "policy_registry_version": [
            trace.manifest.policy_registry_version
        ] * len(trace.request_id),
        "index_version": [trace.manifest.index_version] * len(trace.request_id),
        "fid_version": [trace.manifest.fid_version] * len(trace.request_id),
        "user_event_counts": _fixed_list(context.user_event_counts),
        "user_surface_counts": _fixed_list(context.user_surface_counts),
        "history_item_id": _fixed_list(context.history_item_id),
        "history_event_time": _fixed_list(context.history_event_time),
        "history_ingest_time": _fixed_list(context.history_ingest_time),
    }
    if snapshot.layer_assignment is not None:
        data["layer_cells"] = _fixed_list(
            snapshot.layer_assignment.cell_by_layer
        )
        data["layer_probabilities"] = _fixed_list(
            snapshot.layer_assignment.probability_by_layer
        )
        data["layer_names"] = [
            list(snapshot.layer_assignment.layer_names)
        ] * len(trace.request_id)
    return pa.table(data)


def _route_table(snapshot: FullFlowSnapshot) -> pa.Table:
    trace, catalog = snapshot.trace, snapshot.catalog
    requests, routes, width = trace.route_item_id.shape
    request = trace.request_id[:, None, None].expand(requests, routes, width)
    route = torch.arange(routes, device=trace.request_id.device)[None, :, None]
    route = route.expand(requests, routes, width)
    rank = torch.arange(width, device=trace.request_id.device)[None, None, :]
    rank = rank.expand(requests, routes, width) + 1
    valid = trace.route_valid
    route_id = _numpy(route[valid])
    names = np.asarray(trace.manifest.route_names, dtype=object)[route_id]
    item = trace.route_item_id[valid]
    return pa.table({
        "request_id": _numpy(request[valid]),
        "route_id": route_id,
        "route_name": names,
        "route_rank": _numpy(rank[valid]),
        "item_id": _numpy(item),
        "content_kind": _numpy(catalog.content_kind[item]),
        "creator_id": _numpy(catalog.creator_id[item]),
        "country": _numpy(catalog.country[item]),
        "region": _numpy(catalog.region[item]),
        "publish_time": _numpy(catalog.publish_time[item]),
        "product_id": _numpy(catalog.product_id[item]),
        "poi_id": _numpy(catalog.poi_id[item]),
        "route_score": _numpy(trace.route_score[valid]),
        "candidate_valid": np.ones(int(valid.sum()), dtype=np.bool_),
        "recall_version_id": _numpy(
            trace.recall_version_id[:, None, None]
            .expand(requests, routes, width)[valid]
        ),
        "catalog_version": [trace.manifest.catalog_version] * int(valid.sum()),
    })


def _candidate_table(snapshot: FullFlowSnapshot) -> pa.Table:
    trace, catalog = snapshot.trace, snapshot.catalog
    valid = trace.recall_item_id >= 0
    coarse, coarse_rank, coarse_score = _stage_membership(
        trace.recall_item_id, trace.coarse_item_id, trace.coarse_score,
    )
    fine, fine_rank, fine_score = _stage_membership(
        trace.recall_item_id, trace.fine_item_id, trace.fine_score,
    )
    exposed, exposed_rank, _ = _stage_membership(
        trace.recall_item_id,
        trace.exposed_item_id,
        trace.exposure_probability,
    )
    drop = np.full(trace.recall_item_id.shape, "exposed", dtype=object)
    drop[~_numpy(coarse)] = "coarse_filter"
    drop[_numpy(coarse & ~fine)] = "fine_filter"
    drop[_numpy(fine & ~exposed)] = "mixer_drop"
    recall_rank = torch.arange(
        trace.recall_item_id.shape[1], device=trace.request_id.device,
    )[None].expand_as(trace.recall_item_id) + 1
    request = trace.request_id[:, None].expand_as(trace.recall_item_id)
    item = trace.recall_item_id[valid]
    request_time = trace.event_time[:, None].expand_as(
        trace.recall_item_id
    )[valid]
    exposed_position = torch.where(
        exposed,
        exposed_rank - 1,
        torch.full_like(exposed_rank, -1),
    )
    return pa.table({
        "request_id": _numpy(request[valid]),
        "item_id": _numpy(item),
        "content_kind": _numpy(catalog.content_kind[item]),
        "creator_id": _numpy(catalog.creator_id[item]),
        "country": _numpy(catalog.country[item]),
        "region": _numpy(catalog.region[item]),
        "publish_time": _numpy(catalog.publish_time[item]),
        "content_age": _numpy((request_time - catalog.publish_time[item]).clamp_min(0)),
        "product_id": _numpy(catalog.product_id[item]),
        "poi_id": _numpy(catalog.poi_id[item]),
        "route_bits": _numpy(trace.recall_route_id[valid]),
        "recall_rank": _numpy(recall_rank[valid]),
        "recall_score": _numpy(trace.recall_score[valid]),
        "sampling_probability": _numpy(
            trace.recall_sampling_probability[valid]
        ),
        "coarse_pass": _numpy(coarse[valid]),
        "coarse_rank": _numpy(coarse_rank[valid]),
        "coarse_score": _numpy(coarse_score[valid]),
        "fine_pass": _numpy(fine[valid]),
        "fine_rank": _numpy(fine_rank[valid]),
        "fine_score": _numpy(fine_score[valid]),
        "exposed": _numpy(exposed[valid]),
        "exposed_position": _numpy(exposed_position[valid]),
        "drop_stage": drop[_numpy(valid)],
    })


def _event_table(snapshot: FullFlowSnapshot) -> pa.Table:
    events = snapshot.events
    data = {
        field.name: _numpy(getattr(events, field.name))
        for field in fields(events)
    }
    event_type = data["event_type"]
    data["event_name"] = np.asarray(
        [EventType(int(value)).name.lower() for value in event_type],
        dtype=object,
    )
    return pa.table(data)


def _label_table(snapshot: FullFlowSnapshot) -> pa.Table:
    fine = snapshot.samples.fine
    requests, items, tasks = fine.labels.shape
    request = fine.request_id[:, None, None].expand(requests, items, tasks)
    item = fine.item_id[:, :, None].expand(requests, items, tasks)
    task = torch.arange(tasks, device=fine.labels.device)[None, None, :]
    task = task.expand(requests, items, tasks)
    maturity_ticks = torch.tensor(
        fine.task_maturity_ticks, device=fine.labels.device,
    )[None, None, :].expand(requests, items, tasks)
    maturity_time = fine.request_time[:, None, None] + maturity_ticks
    valid = item >= 0
    mature = fine.label_mask
    value = torch.where(
        mature, fine.labels, torch.full_like(fine.labels, torch.nan),
    )
    watermark = torch.full_like(request, snapshot.samples.event_watermark)
    censor = np.full(fine.labels.shape, "observed", dtype=object)
    censor[_numpy(~mature & (watermark < maturity_time))] = "label_not_mature"
    censor[_numpy(~mature & (watermark >= maturity_time))] = "not_applicable"
    task_id = _numpy(task[valid])
    names = np.asarray(fine.task_names, dtype=object)[task_id]
    return pa.table({
        "request_id": _numpy(request[valid]),
        "item_id": _numpy(item[valid]),
        "task_id": task_id,
        "task_name": names,
        "label_value": _numpy(value[valid]),
        "label_mask": _numpy(mature[valid]),
        "maturity_time": _numpy(maturity_time[valid]),
        "event_watermark": _numpy(watermark[valid]),
        "censor_reason": censor[_numpy(valid)],
    })


def _example_table(snapshot: FullFlowSnapshot) -> pa.Table:
    samples = snapshot.samples
    tables = []
    recall = samples.recall
    positive_rows = len(recall.request_id)
    tables.append(pa.table({
        "request_id": _numpy(recall.request_id),
        "item_id": _numpy(recall.positive_item_id),
        "authority": ["recall"] * positive_rows,
        "role": ["positive"] * positive_rows,
        "ordinal": np.zeros(positive_rows, dtype=np.int64),
        "sampling_probability": np.ones(positive_rows, dtype=np.float32),
        "label_value": _numpy(recall.positive_strength),
        "label_mask": np.ones(positive_rows, dtype=np.bool_),
        "teacher_score": np.full(positive_rows, np.nan, dtype=np.float32),
        "teacher_mask": np.zeros(positive_rows, dtype=np.bool_),
    }))
    neg_valid = recall.negative_item_id >= 0
    neg_rank = torch.arange(
        recall.negative_item_id.shape[1], device=recall.request_id.device,
    )[None].expand_as(recall.negative_item_id)
    neg_request = recall.request_id[:, None].expand_as(recall.negative_item_id)
    negative_rows = int(neg_valid.sum())
    tables.append(pa.table({
        "request_id": _numpy(neg_request[neg_valid]),
        "item_id": _numpy(recall.negative_item_id[neg_valid]),
        "authority": ["recall"] * negative_rows,
        "role": ["negative"] * negative_rows,
        "ordinal": _numpy(neg_rank[neg_valid]),
        "sampling_probability": _numpy(
            recall.negative_sampling_probability[neg_valid]
        ),
        "label_value": np.zeros(negative_rows, dtype=np.float32),
        "label_mask": np.ones(negative_rows, dtype=np.bool_),
        "teacher_score": np.full(negative_rows, np.nan, dtype=np.float32),
        "teacher_mask": np.zeros(negative_rows, dtype=np.bool_),
    }))
    coarse = samples.coarse
    coarse_valid = coarse.item_id >= 0
    coarse_rank = torch.arange(
        coarse.item_id.shape[1], device=coarse.item_id.device,
    )[None].expand_as(coarse.item_id)
    coarse_request = coarse.request_id[:, None].expand_as(coarse.item_id)
    coarse_rows = int(coarse_valid.sum())
    tables.append(pa.table({
        "request_id": _numpy(coarse_request[coarse_valid]),
        "item_id": _numpy(coarse.item_id[coarse_valid]),
        "authority": ["coarse"] * coarse_rows,
        "role": ["candidate"] * coarse_rows,
        "ordinal": _numpy(coarse_rank[coarse_valid]),
        "sampling_probability": _numpy(
            coarse.sampling_probability[coarse_valid]
        ),
        "label_value": _numpy(coarse.hard_label[coarse_valid]),
        "label_mask": _numpy(coarse.hard_label_mask[coarse_valid]),
        "teacher_score": _numpy(coarse.teacher_score[coarse_valid]),
        "teacher_mask": _numpy(coarse.teacher_mask[coarse_valid]),
    }))
    fine = samples.fine
    fine_valid = fine.item_id >= 0
    fine_rank = torch.arange(
        fine.item_id.shape[1], device=fine.item_id.device,
    )[None].expand_as(fine.item_id)
    fine_request = fine.request_id[:, None].expand_as(fine.item_id)
    fine_rows = int(fine_valid.sum())
    fine_value = torch.where(
        fine.label_mask, fine.labels, torch.zeros_like(fine.labels),
    ).sum(dim=2)
    tables.append(pa.table({
        "request_id": _numpy(fine_request[fine_valid]),
        "item_id": _numpy(fine.item_id[fine_valid]),
        "authority": ["fine"] * fine_rows,
        "role": ["impression"] * fine_rows,
        "ordinal": _numpy(fine_rank[fine_valid]),
        "sampling_probability": _numpy(
            fine.exposure_probability[fine_valid]
        ),
        "label_value": _numpy(fine_value[fine_valid]),
        "label_mask": _numpy(fine.label_mask.any(dim=2)[fine_valid]),
        "teacher_score": _numpy(fine.served_score[fine_valid]),
        "teacher_mask": np.ones(fine_rows, dtype=np.bool_),
    }))
    result = pa.concat_tables(tables)
    rows = len(result)
    return result.append_column(
        "feature_version",
        pa.array([snapshot.trace.manifest.feature_version] * rows),
    ).append_column(
        "example_watermark",
        pa.array(np.full(rows, samples.event_watermark, dtype=np.int64)),
    )


def _checkpoint_table(records: tuple[CheckpointRecord, ...]) -> pa.Table:
    schema = pa.schema((
        ("created_time", pa.int64()),
        ("lane", pa.string()),
        ("model_name", pa.string()),
        ("checkpoint_version", pa.string()),
        ("data_watermark", pa.int64()),
        ("sample_manifest", pa.string()),
        ("feature_version", pa.string()),
        ("fid_version", pa.string()),
        ("index_version", pa.string()),
        ("validation_status", pa.string()),
        ("publish_state", pa.string()),
        ("fallback_version", pa.string()),
    ))
    if not records:
        return pa.Table.from_batches([], schema=schema)
    return pa.Table.from_pylist([
        {field.name: getattr(record, field.name) for field in fields(record)}
        for record in records
    ], schema=schema)


def iter_full_flow_tables(snapshot: FullFlowSnapshot):
    """Yield one table at a time so scale writes do not retain all Arrow copies."""
    builders = (
        ("v4_request_log", _request_table),
        ("v4_route_candidate_log", _route_table),
        ("v4_candidate_decision_log", _candidate_table),
        ("v4_event_log", _event_table),
        ("v4_mature_label_log", _label_table),
        ("v4_training_example_log", _example_table),
    )
    if tuple(name for name, _ in builders) + ("v4_checkpoint_log",) != TABLE_NAMES:
        raise AssertionError("full-flow table registry order changed")
    for name, build in builders:
        yield name, build(snapshot)
    yield "v4_checkpoint_log", _checkpoint_table(snapshot.checkpoints)


def build_full_flow_tables(snapshot: FullFlowSnapshot) -> dict[str, pa.Table]:
    """In-memory view for small tests and interactive diagnostics."""
    return dict(iter_full_flow_tables(snapshot))
