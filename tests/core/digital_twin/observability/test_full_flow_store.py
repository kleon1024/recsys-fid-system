from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from fid_lab.simulation.digital_twin.observability import (
    FullFlowSnapshot,
    TABLE_NAMES,
    build_full_flow_fixture,
    build_full_flow_tables,
    materialize_full_flow,
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
    stages = {
        row[0]
        for row in connection.execute(
            "SELECT drop_stage FROM v4_stage_distribution"
        ).fetchall()
    }
    assert {"coarse_filter", "fine_filter", "mixer_drop", "exposed"} <= stages
