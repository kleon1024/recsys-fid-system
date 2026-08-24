"""Explicit analytical failure injection; never a training-data source."""

from __future__ import annotations

import pyarrow as pa


def _append_changed_row(
    table: pa.Table,
    changes: dict[str, object],
) -> pa.Table:
    if not len(table):
        raise ValueError("cannot seed a failure from an empty table")
    row = table.slice(0, 1).to_pylist()[0]
    row.update(changes)
    return pa.concat_tables((
        table,
        pa.Table.from_pylist((row,), schema=table.schema),
    ))


def seed_diagnostic_failures(
    tables: dict[str, pa.Table],
) -> dict[str, pa.Table]:
    """Add one recall miss, request orphan and rejected checkpoint."""
    result = dict(tables)
    candidate = result["v4_candidate_decision_log"]
    missing_item = int(candidate["item_id"].to_numpy().max()) + 1
    examples = result["v4_training_example_log"]
    result["v4_training_example_log"] = _append_changed_row(examples, {
        "item_id": missing_item,
        "authority": "recall",
        "role": "positive",
        "label_value": 1.0,
        "label_mask": True,
        "teacher_score": None,
        "teacher_mask": False,
    })
    requests = result["v4_request_log"]
    orphan_request = int(requests["request_id"].to_numpy().max()) + 1_000_003
    events = result["v4_event_log"]
    result["v4_event_log"] = _append_changed_row(events, {
        "event_id": int(events["event_id"].to_numpy().max()) + 1,
        "event_type": 7,
        "event_name": "click",
        "request_id": orphan_request,
        "user_id": int(requests["user_id"][0].as_py()),
    })
    checkpoint = result["v4_checkpoint_log"]
    result["v4_checkpoint_log"] = _append_changed_row(checkpoint, {
        "created_time": int(checkpoint["created_time"][0].as_py()) + 1,
        "lane": "candidate",
        "checkpoint_version": "checkpoint-invalid",
        "index_version": "index-mismatch",
        "validation_status": "reject",
        "publish_state": "rejected",
        "fallback_version": str(
            checkpoint["checkpoint_version"][0].as_py()
        ),
    })
    return result
