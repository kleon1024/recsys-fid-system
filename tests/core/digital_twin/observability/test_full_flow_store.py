from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from fid_lab.simulation.digital_twin.observability import (
    FullFlowSnapshot,
    FullFlowFixtureConfig,
    TABLE_NAMES,
    append_full_flow_partition,
    build_full_flow_fixture,
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
    labels = tables["v4_mature_label_log"].to_pandas()
    assert labels.loc[~labels.label_mask, "label_value"].isna().all()
    examples = tables["v4_training_example_log"].to_pandas()
    assert set(examples.authority) == {"recall", "coarse", "fine"}


def test_parquet_manifest_is_content_bound_and_replayable(tmp_path):
    snapshot = _snapshot()
    manifest = materialize_full_flow(snapshot, tmp_path)
    persisted = json.loads((tmp_path / "manifest.json").read_text())
    assert persisted["schema"] == "digital-twin-full-flow-v1"
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
