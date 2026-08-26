"""Vectorized Arrow tables derived from one factual v4 snapshot."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pyarrow as pa
import torch

from ..contracts import EventType
from ..platform.lifecycle import ContentLifecycle, post_content_mask
from ..platform.routes import FEED_ROUTE_NAMES
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

LIFECYCLE_NAMES = np.asarray(
    [state.name.lower() for state in ContentLifecycle], dtype=object,
)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _fixed_list(value: torch.Tensor) -> pa.FixedSizeListArray:
    if value.ndim != 2:
        raise ValueError("fixed-list source must be two-dimensional")
    flat = pa.array(_numpy(value).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, value.shape[1])


def _variable_list(value: torch.Tensor, value_type: pa.DataType) -> pa.ListArray:
    if value.ndim != 2:
        raise ValueError("variable-list source must be two-dimensional")
    rows, width = value.shape
    offsets = pa.array(
        np.arange(rows + 1, dtype=np.int32) * width, type=pa.int32(),
    )
    values = pa.array(_numpy(value).reshape(-1), type=value_type)
    return pa.ListArray.from_arrays(offsets, values)


def _empty_lists(rows: int, value_type: pa.DataType) -> pa.ListArray:
    return pa.ListArray.from_arrays(
        pa.array(np.zeros(rows + 1, dtype=np.int32), type=pa.int32()),
        pa.array([], type=value_type),
    )


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
        "user_creator_id": _numpy(trace.user_creator_id),
        "experiment_cell": _numpy(trace.experiment_cell),
        "assignment_probability": _numpy(trace.assignment_probability),
        "selection_policy_kind": _numpy(trace.selection_policy_kind),
        "exploration_rate": _numpy(trace.exploration_rate),
        "slate_log_probability": _numpy(trace.slate_log_probability),
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
        "lifecycle_version": [
            trace.manifest.lifecycle_version
        ] * len(trace.request_id),
        "feature_manifest_hash": [
            trace.manifest.feature_manifest_hash
        ] * len(trace.request_id),
        "user_event_counts": _fixed_list(context.user_event_counts),
        "user_surface_counts": _fixed_list(context.user_surface_counts),
        "history_item_id": _fixed_list(context.history_item_id),
        "history_event_type": _fixed_list(context.history_event_type),
        "history_surface": _fixed_list(context.history_surface),
        "history_duration_ms": _fixed_list(context.history_duration_ms),
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
    state = snapshot.projection.state
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
    lifecycle = trace.route_lifecycle_id[valid]
    post = torch.where(
        post_content_mask(catalog.content_kind[item]),
        item,
        torch.full_like(item, -1),
    )
    feed_route = np.isin(names, np.asarray(FEED_ROUTE_NAMES, dtype=object))
    return pa.table({
        "request_id": _numpy(request[valid]),
        "route_id": route_id,
        "route_name": names,
        "route_rank": _numpy(rank[valid]),
        "item_id": _numpy(item),
        "post_id": _numpy(post),
        "topic_id": _numpy(catalog.topic_id[item]),
        "content_kind": _numpy(catalog.content_kind[item]),
        "creator_id": _numpy(state.item_creator_id[item]),
        "country": _numpy(state.item_country[item]),
        "region": _numpy(state.item_region[item]),
        "publish_time": _numpy(state.item_publish_time[item]),
        "product_id": _numpy(state.item_product_id[item]),
        "poi_id": _numpy(state.item_poi_id[item]),
        "lifecycle_id": _numpy(lifecycle),
        "lifecycle_name": LIFECYCLE_NAMES[_numpy(lifecycle)],
        "route_admission_reason": np.where(
            feed_route, "feed_lifecycle", "business_surface_contract",
        ),
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
    state = snapshot.projection.state
    valid = trace.recall_item_id >= 0
    coarse, coarse_rank, _ = _stage_membership(
        trace.recall_item_id,
        trace.coarse_item_id,
        trace.coarse_selected_score,
    )
    fine, fine_rank, _ = _stage_membership(
        trace.recall_item_id,
        trace.fine_item_id,
        trace.fine_selected_score,
    )
    _, _, fine_input_score = _stage_membership(
        trace.recall_item_id,
        trace.coarse_item_id,
        trace.fine_input_score,
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
    lifecycle = trace.recall_lifecycle_id[valid]
    post = torch.where(
        post_content_mask(catalog.content_kind[item]),
        item,
        torch.full_like(item, -1),
    )
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
        "post_id": _numpy(post),
        "topic_id": _numpy(catalog.topic_id[item]),
        "content_kind": _numpy(catalog.content_kind[item]),
        "creator_id": _numpy(state.item_creator_id[item]),
        "country": _numpy(state.item_country[item]),
        "region": _numpy(state.item_region[item]),
        "publish_time": _numpy(state.item_publish_time[item]),
        "content_age": _numpy(
            (request_time - state.item_publish_time[item]).clamp_min(0)
        ),
        "product_id": _numpy(state.item_product_id[item]),
        "poi_id": _numpy(state.item_poi_id[item]),
        "lifecycle_id": _numpy(lifecycle),
        "lifecycle_name": LIFECYCLE_NAMES[_numpy(lifecycle)],
        "route_bits": _numpy(trace.recall_route_id[valid]),
        "recall_rank": _numpy(recall_rank[valid]),
        "recall_score": _numpy(trace.recall_score[valid]),
        "sampling_probability": _numpy(
            trace.recall_sampling_probability[valid]
        ),
        "coarse_pass": _numpy(coarse[valid]),
        "coarse_rank": _numpy(coarse_rank[valid]),
        "coarse_score": _numpy(trace.coarse_input_score[valid]),
        "coarse_admission_probability": _numpy(
            trace.coarse_admission_probability[valid]
        ),
        "fine_pass": _numpy(fine[valid]),
        "fine_rank": _numpy(fine_rank[valid]),
        "fine_score": _numpy(fine_input_score[valid]),
        "fine_admission_probability": _numpy(
            trace.fine_admission_probability[valid]
        ),
        "candidate_exposure_probability": _numpy(
            trace.candidate_exposure_probability[valid]
        ),
        "exposed": _numpy(exposed[valid]),
        "exposed_position": _numpy(exposed_position[valid]),
        "candidate_dense_features": _variable_list(
            trace.candidate_dense_features[valid], pa.float32(),
        ),
        "candidate_sparse_fids": _variable_list(
            trace.candidate_sparse_fids[valid], pa.int64(),
        ),
        "candidate_sparse_buckets": _variable_list(
            trace.candidate_sparse_buckets[valid], pa.int64(),
        ),
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
    maturity_time = fine.label_maturity_time
    valid = item >= 0
    applicable = fine.label_applicable
    mature = fine.label_mature
    observed = fine.label_mask
    value = torch.where(
        observed, fine.labels, torch.full_like(fine.labels, torch.nan),
    )
    watermark = torch.full_like(request, snapshot.samples.event_watermark)
    censor = np.full(fine.labels.shape, "observed", dtype=object)
    censor[_numpy(applicable & ~mature)] = "label_not_mature"
    censor[_numpy(~applicable)] = "not_applicable"
    task_id = _numpy(task[valid])
    names = np.asarray(fine.task_names, dtype=object)[task_id]
    return pa.table({
        "request_id": _numpy(request[valid]),
        "item_id": _numpy(item[valid]),
        "task_id": task_id,
        "task_name": names,
        "label_value": _numpy(value[valid]),
        "label_applicable": _numpy(applicable[valid]),
        "label_mature": _numpy(mature[valid]),
        "label_mask": _numpy(observed[valid]),
        "maturity_time": _numpy(maturity_time[valid]),
        "event_watermark": _numpy(watermark[valid]),
        "censor_reason": censor[_numpy(valid)],
    })


def _sample_lineage_defaults(rows: int) -> dict[str, object]:
    return {
        "sampling_source": np.full(rows, -1, dtype=np.int64),
        "sampling_expected_count": np.full(rows, np.nan, dtype=np.float32),
        "false_negative_mask": np.zeros(rows, dtype=np.bool_),
        "negative_observed": np.zeros(rows, dtype=np.bool_),
        "route_id": np.full(rows, -1, dtype=np.int64),
        "recall_score": np.full(rows, np.nan, dtype=np.float32),
        "coarse_rank": np.full(rows, -1, dtype=np.int64),
        "teacher_rank": np.full(rows, -1, dtype=np.int64),
        "conflict_mask": np.zeros(rows, dtype=np.bool_),
        "joint_logging_probability": np.full(
            rows, np.nan, dtype=np.float32,
        ),
        "factual_exposure_probability": np.zeros(rows, dtype=np.float32),
        "candidate_exposure_probability": np.zeros(rows, dtype=np.float32),
        "ope_supported": np.zeros(rows, dtype=np.bool_),
        "randomized_support": np.zeros(rows, dtype=np.bool_),
        "coarse_admitted": np.zeros(rows, dtype=np.bool_),
        "fine_admitted": np.zeros(rows, dtype=np.bool_),
        "exposed": np.zeros(rows, dtype=np.bool_),
        "selection_policy_kind": np.full(rows, -1, dtype=np.int64),
        "exploration_rate": np.zeros(rows, dtype=np.float32),
        "slate_log_probability": np.full(rows, np.nan, dtype=np.float32),
        "user_id": np.full(rows, -1, dtype=np.int64),
        "surface": np.full(rows, -1, dtype=np.int64),
        "request_time": np.full(rows, -1, dtype=np.int64),
        "position": np.full(rows, -1, dtype=np.int64),
        "served_checkpoint_id": np.full(rows, -1, dtype=np.int64),
        "feature_manifest_hash": np.full(rows, "", dtype=object),
        "dense_features": _empty_lists(rows, pa.float32()),
        "sparse_fids": _empty_lists(rows, pa.int64()),
        "sparse_buckets": _empty_lists(rows, pa.int64()),
        "task_label_values": _empty_lists(rows, pa.float32()),
        "task_label_masks": _empty_lists(rows, pa.bool_()),
        "task_label_applicable": _empty_lists(rows, pa.bool_()),
        "task_label_mature": _empty_lists(rows, pa.bool_()),
    }


def _recall_example_tables(recall) -> tuple[pa.Table, pa.Table]:
    positive_rows = len(recall.request_id)
    positive = {
        "request_id": _numpy(recall.request_id),
        "item_id": _numpy(recall.positive_item_id),
        "authority": ["recall"] * positive_rows,
        "role": ["positive"] * positive_rows,
        "ordinal": np.zeros(positive_rows, dtype=np.int64),
        "sampling_probability": _numpy(
            recall.positive_proposal_probability
        ),
        "label_value": _numpy(recall.positive_strength),
        "label_mask": np.ones(positive_rows, dtype=np.bool_),
        "teacher_score": np.full(positive_rows, np.nan, dtype=np.float32),
        "teacher_mask": np.zeros(positive_rows, dtype=np.bool_),
        **_sample_lineage_defaults(positive_rows),
    }
    positive["route_id"] = _numpy(recall.positive_route_id)
    neg_valid = recall.negative_item_id >= 0
    neg_rank = torch.arange(
        recall.negative_item_id.shape[1], device=recall.request_id.device,
    )[None].expand_as(recall.negative_item_id)
    neg_request = recall.request_id[:, None].expand_as(recall.negative_item_id)
    negative_rows = int(neg_valid.sum())
    negative = {
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
        **_sample_lineage_defaults(negative_rows),
    }
    negative["sampling_source"] = _numpy(
        recall.negative_source[neg_valid]
    )
    negative["sampling_expected_count"] = _numpy(
        recall.negative_expected_count[neg_valid]
    )
    negative["false_negative_mask"] = _numpy(
        recall.negative_false_negative_mask[neg_valid]
    )
    negative["negative_observed"] = _numpy(
        recall.negative_observed[neg_valid]
    )
    return pa.table(positive), pa.table(negative)


def _coarse_example_table(coarse) -> pa.Table:
    coarse_valid = coarse.item_id >= 0
    coarse_rank = torch.arange(
        coarse.item_id.shape[1], device=coarse.item_id.device,
    )[None].expand_as(coarse.item_id)
    coarse_request = coarse.request_id[:, None].expand_as(coarse.item_id)
    coarse_rows = int(coarse_valid.sum())
    coarse_data = {
        "request_id": _numpy(coarse_request[coarse_valid]),
        "item_id": _numpy(coarse.item_id[coarse_valid]),
        "authority": ["coarse"] * coarse_rows,
        "role": ["candidate"] * coarse_rows,
        "ordinal": _numpy(coarse_rank[coarse_valid]),
        "sampling_probability": _numpy(
            coarse.admission_probability[coarse_valid]
        ),
        "label_value": _numpy(coarse.hard_label[coarse_valid]),
        "label_mask": _numpy(coarse.hard_label_mask[coarse_valid]),
        "teacher_score": _numpy(coarse.teacher_score[coarse_valid]),
        "teacher_mask": _numpy(coarse.teacher_mask[coarse_valid]),
        **_sample_lineage_defaults(coarse_rows),
    }
    coarse_data["route_id"] = _numpy(coarse.route_id[coarse_valid])
    coarse_data["recall_score"] = _numpy(
        coarse.recall_score[coarse_valid]
    )
    coarse_data["coarse_rank"] = _numpy(coarse.coarse_rank[coarse_valid])
    coarse_data["coarse_admitted"] = _numpy(
        coarse.coarse_admitted[coarse_valid]
    )
    coarse_data["teacher_rank"] = _numpy(coarse.teacher_rank[coarse_valid])
    coarse_data["conflict_mask"] = _numpy(
        coarse.conflict_mask[coarse_valid]
    )
    return pa.table(coarse_data)


def _fine_example_table(fine) -> pa.Table:
    fine_valid = fine.item_id >= 0
    fine_rank = torch.arange(
        fine.item_id.shape[1], device=fine.item_id.device,
    )[None].expand_as(fine.item_id)
    fine_request = fine.request_id[:, None].expand_as(fine.item_id)
    fine_rows = int(fine_valid.sum())
    fine_data = {
        "request_id": _numpy(fine_request[fine_valid]),
        "item_id": _numpy(fine.item_id[fine_valid]),
        "authority": ["fine"] * fine_rows,
        "role": ["candidate"] * fine_rows,
        "ordinal": _numpy(fine_rank[fine_valid]),
        "sampling_probability": _numpy(
            fine.candidate_exposure_probability[fine_valid]
        ),
        "label_value": np.full(fine_rows, np.nan, dtype=np.float32),
        "label_mask": np.zeros(fine_rows, dtype=np.bool_),
        "teacher_score": _numpy(fine.served_score[fine_valid]),
        "teacher_mask": np.ones(fine_rows, dtype=np.bool_),
        **_sample_lineage_defaults(fine_rows),
    }
    fine_data["route_id"] = _numpy(fine.recall_route_id[fine_valid])
    fine_data["recall_score"] = _numpy(fine.recall_score[fine_valid])
    fine_data["joint_logging_probability"] = _numpy(
        fine.joint_logging_probability[fine_valid]
    )
    fine_data["factual_exposure_probability"] = _numpy(
        fine.exposure_probability[fine_valid]
    )
    fine_data["candidate_exposure_probability"] = _numpy(
        fine.candidate_exposure_probability[fine_valid]
    )
    fine_data["randomized_support"] = _numpy(
        fine.randomized_support[fine_valid]
    )
    fine_data["fine_admitted"] = _numpy(fine.fine_admitted[fine_valid])
    fine_data["exposed"] = _numpy(fine.exposed[fine_valid])
    fine_data["user_id"] = _numpy(
        fine.user_id[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["surface"] = _numpy(
        fine.surface[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["request_time"] = _numpy(
        fine.request_time[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["position"] = _numpy(fine.position[fine_valid])
    fine_data["selection_policy_kind"] = _numpy(
        fine.selection_policy_kind[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["exploration_rate"] = _numpy(
        fine.exploration_rate[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["slate_log_probability"] = _numpy(
        fine.slate_log_probability[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["served_checkpoint_id"] = _numpy(
        fine.fine_version_id[:, None].expand_as(fine.item_id)[fine_valid]
    )
    fine_data["feature_manifest_hash"] = np.full(
        fine_rows, fine.feature_manifest_hash, dtype=object,
    )
    fine_data["dense_features"] = _variable_list(
        fine.dense_features[fine_valid], pa.float32(),
    )
    fine_data["sparse_fids"] = _variable_list(
        fine.sparse_fids[fine_valid], pa.int64(),
    )
    fine_data["sparse_buckets"] = _variable_list(
        fine.sparse_buckets[fine_valid], pa.int64(),
    )
    fine_data["task_label_values"] = _variable_list(
        fine.labels[fine_valid], pa.float32(),
    )
    fine_data["task_label_masks"] = _variable_list(
        fine.label_mask[fine_valid], pa.bool_(),
    )
    fine_data["task_label_applicable"] = _variable_list(
        fine.label_applicable[fine_valid], pa.bool_(),
    )
    fine_data["task_label_mature"] = _variable_list(
        fine.label_mature[fine_valid], pa.bool_(),
    )
    return pa.table(fine_data)


def _example_table(snapshot: FullFlowSnapshot) -> pa.Table:
    samples = snapshot.samples
    result = pa.concat_tables((
        *_recall_example_tables(samples.recall),
        _coarse_example_table(samples.coarse),
        _fine_example_table(samples.fine),
    ))
    rows = len(result)
    return result.append_column(
        "feature_version",
        pa.array([snapshot.trace.manifest.feature_version] * rows),
    ).append_column(
        "catalog_version",
        pa.array([snapshot.trace.manifest.catalog_version] * rows),
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
        ("serving_version_id", pa.int64()),
        ("artifact_sha256", pa.string()),
        ("compatibility_hash", pa.string()),
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
