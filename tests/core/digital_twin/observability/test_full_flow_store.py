from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import duckdb
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from fid_lab.simulation.digital_twin.observability import (
    FullFlowSnapshot,
    FullFlowFixtureConfig,
    FULL_FLOW_SCHEMA_VERSION,
    TABLE_NAMES,
    append_full_flow_partition,
    build_full_flow_fixture,
    build_full_flow_fixtures,
    build_full_flow_tables,
    materialize_full_flow,
    open_full_flow_dataset,
    seed_diagnostic_failures,
    verify_full_flow_dataset,
)
from fid_lab.simulation.digital_twin.observability.diagnostics import (
    install_diagnostics,
    register_full_flow,
    request_case,
)


ROOT = Path(__file__).resolve().parents[4]


def _snapshot() -> FullFlowSnapshot:
    return build_full_flow_fixture()


def test_full_flow_tables_share_one_request_and_sample_closure():
    snapshot = _snapshot()
    tables = build_full_flow_tables(snapshot)
    assert tuple(tables) == TABLE_NAMES
    assert len(tables["v4_request_log"]) == len(snapshot.trace.request_id)
    assert len(tables["v4_route_candidate_log"]) > 0
    assert len(tables["v4_candidate_decision_log"]) > 0
    assert len(tables["v4_event_log"]) == len(snapshot.events.event_id)
    assert len(tables["v4_checkpoint_log"]) == 1
    request = tables["v4_request_log"].to_pandas()
    assert request.index_version.eq("observable-index-t0").all()
    assert request.fid_version.eq("fid-v2").all()
    assert request.lifecycle_version.eq("content-lifecycle-v1").all()
    assert request.feature_manifest_hash.str.len().eq(64).all()
    assert {
        "history_event_type", "history_surface", "history_duration_ms",
    } <= set(request.columns)
    routes = tables["v4_route_candidate_log"].to_pandas()
    assert routes.lifecycle_name.notna().all()
    assert routes.route_admission_reason.isin({
        "feed_lifecycle", "business_surface_contract",
    }).all()
    assert (routes.loc[routes.post_id >= 0, "post_id"] == routes.loc[
        routes.post_id >= 0, "item_id"
    ]).all()
    labels = tables["v4_mature_label_log"].to_pandas()
    assert labels.loc[~labels.label_mask, "label_value"].isna().all()
    assert (labels.label_mask == (
        labels.label_applicable & labels.label_mature
    )).all()
    examples = tables["v4_training_example_log"].to_pandas()
    assert set(examples.authority) == {"recall", "coarse", "fine"}
    negatives = examples[
        (examples.authority == "recall") & (examples.role == "negative")
    ]
    assert negatives.sampling_expected_count.notna().all()
    assert (negatives.sampling_expected_count > 0).all()
    coarse = examples[examples.authority == "coarse"]
    assert (coarse.loc[coarse.teacher_mask, "teacher_rank"] > 0).all()
    fine = examples[examples.authority == "fine"]
    assert fine.joint_logging_probability.notna().all()
    assert not fine.randomized_support.any()
    assert fine.label_value.isna().all()
    assert not fine.label_mask.any()
    assert fine.feature_manifest_hash.str.len().eq(64).all()
    expected = snapshot.samples.fine
    valid = expected.item_id >= 0
    assert fine.iloc[0].dense_features.tolist() == (
        expected.dense_features[valid][0].tolist()
    )
    assert fine.iloc[0].sparse_fids.tolist() == (
        expected.sparse_fids[valid][0].tolist()
    )
    assert fine.iloc[0].task_label_values.tolist() == (
        expected.labels[valid][0].tolist()
    )
    assert fine.iloc[0].task_label_masks.tolist() == (
        expected.label_mask[valid][0].tolist()
    )


def test_route_lifecycle_is_request_time_not_post_response_projection():
    snapshot = _snapshot()
    valid = snapshot.trace.route_valid
    item = snapshot.trace.route_item_id[valid]
    request_time = snapshot.trace.route_lifecycle_id[valid]
    post_response = snapshot.projection.state.item_lifecycle[item]
    assert (request_time != post_response).any()
    table = build_full_flow_tables(snapshot)["v4_route_candidate_log"]
    assert table["lifecycle_id"].to_pylist() == request_time.tolist()


def test_parquet_manifest_is_content_bound_and_replayable(tmp_path):
    snapshot = _snapshot()
    manifest = materialize_full_flow(snapshot, tmp_path)
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted["schema"] == FULL_FLOW_SCHEMA_VERSION
    assert set(persisted["tables"]) == set(TABLE_NAMES)
    for name in TABLE_NAMES:
        evidence = manifest["tables"][name]
        table = pq.read_table(tmp_path / evidence["file"])
        assert len(table) == evidence["rows"]
        assert len(evidence["sha256"]) == 64


def test_duckdb_case_and_stage_queries_execute_on_arrow_authority():
    tables = build_full_flow_tables(_snapshot())
    connection = duckdb.connect()
    register_full_flow(connection, tables)
    install_diagnostics(
        connection,
        ROOT / "sql" / "duckdb" / "v4_full_flow_diagnostics.sql",
    )
    request_id = int(tables["v4_request_log"]["request_id"][0].as_py())
    case = request_case(connection, request_id)
    assert len(case) > 0
    assert case["recall_rank"].to_pylist() == sorted(
        case["recall_rank"].to_pylist()
    )
    exposed = case.filter(case["exposed"])
    assert min(exposed["exposed_position"].to_pylist()) == 0
    stages = {
        row[0]
        for row in connection.execute(
            "SELECT drop_stage FROM v4_stage_distribution"
        ).fetchall()
    }
    assert {"coarse_filter", "fine_filter", "mixer_drop", "exposed"} <= stages
    assert connection.execute(
        "SELECT count(*) FROM v4_route_admission_violations"
    ).fetchone()[0] == 0


def test_seeded_failures_are_independently_diagnosed():
    tables = seed_diagnostic_failures(build_full_flow_tables(_snapshot()))
    connection = duckdb.connect()
    register_full_flow(connection, tables)
    install_diagnostics(
        connection,
        ROOT / "sql" / "duckdb" / "v4_full_flow_diagnostics.sql",
    )
    assert connection.execute("SELECT count(*) FROM v4_recall_miss").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM v4_orphan_events").fetchone()[0] == 1
    assert connection.execute(
        "SELECT count(*) FROM v4_checkpoint_health WHERE unhealthy"
    ).fetchone()[0] == 1


def test_partition_bus_resumes_exact_content_and_rejects_key_drift(tmp_path):
    snapshot = _snapshot()
    first = append_full_flow_partition(snapshot, tmp_path, "event_time=0")
    assert first["status"] == "written"
    resumed = append_full_flow_partition(snapshot, tmp_path, "event_time=0")
    assert resumed["status"] == "resumed"
    assert resumed["partition_content_sha256"] == first[
        "partition_content_sha256"
    ]
    second_snapshot = build_full_flow_fixture(FullFlowFixtureConfig(
        logical_time=1,
    ))
    second = append_full_flow_partition(
        second_snapshot, tmp_path, "event_time=1",
    )
    assert second["status"] == "written"
    dataset = verify_full_flow_dataset(tmp_path)
    assert list(dataset["partitions"]) == ["event_time=0", "event_time=1"]
    assert dataset["table_rows"]["v4_request_log"] == (
        len(snapshot.trace.request_id)
        + len(second_snapshot.trace.request_id)
    )
    lazy = open_full_flow_dataset(tmp_path)
    assert lazy["v4_request_log"].count_rows() == dataset[
        "table_rows"
    ]["v4_request_log"]
    event_times = lazy["v4_request_log"].to_table(
        columns=["event_time"],
    )["event_time"].to_pylist()
    assert set(event_times) == {0, 1}
    with pytest.raises(ValueError, match="does not match data"):
        append_full_flow_partition(snapshot, tmp_path, "event_time=2")
    altered_trace = replace(
        snapshot.trace,
        recall_score=snapshot.trace.recall_score + 0.01,
    )
    altered = replace(snapshot, trace=altered_trace)
    with pytest.raises(ValueError, match="different content"):
        append_full_flow_partition(altered, tmp_path, "event_time=0")


def test_multi_tick_posting_cycle_is_queryable_without_reconstructing_world(
    tmp_path,
):
    snapshots = build_full_flow_fixtures(FullFlowFixtureConfig(
        users=256,
        items=4_000,
        scenario="feed_posting_cycle",
    ), ticks=2)
    for logical_time, snapshot in enumerate(snapshots):
        append_full_flow_partition(
            snapshot, tmp_path, f"event_time={logical_time}",
        )
    dataset = open_full_flow_dataset(tmp_path)
    events = dataset["v4_event_log"].to_table()
    published = events.filter(
        pc.equal(events["event_name"], "publish")
    )["post_id"].to_pylist()
    assert published
    routes = dataset["v4_route_candidate_log"].to_table().to_pandas()
    requests = dataset["v4_request_log"].to_table().to_pandas()
    later_request = set(requests.loc[requests.event_time == 1, "request_id"])
    later = routes.loc[routes.request_id.isin(later_request)]
    recalled_published = set(published) & set(later.post_id)
    assert len(recalled_published) >= min(len(set(published)), 16)
    tables = {name: source.to_table() for name, source in dataset.items()}
    connection = duckdb.connect()
    register_full_flow(connection, tables)
    install_diagnostics(
        connection,
        ROOT / "sql" / "duckdb" / "v4_full_flow_diagnostics.sql",
    )
    assert connection.execute(
        "SELECT count(*) FROM v4_route_admission_violations"
    ).fetchone()[0] == 0


def test_partition_verifier_rejects_corrupted_parquet(tmp_path):
    append_full_flow_partition(_snapshot(), tmp_path, "event_time=0")
    path = (
        tmp_path
        / "partitions"
        / "event_time=0"
        / "v4_candidate_decision_log.parquet"
    )
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="table hash mismatch"):
        verify_full_flow_dataset(tmp_path)
