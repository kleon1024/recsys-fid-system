"""Executable DuckDB mirror of the ClickHouse full-flow investigations."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa


DIAGNOSTIC_VIEWS = (
    "v4_request_case",
    "v4_stage_distribution",
    "v4_route_distribution",
    "v4_route_marginal_coverage",
    "v4_candidate_slices",
    "v4_label_maturity",
    "v4_recall_miss",
    "v4_orphan_events",
    "v4_checkpoint_health",
)


def register_full_flow(
    connection: duckdb.DuckDBPyConnection,
    tables: dict[str, pa.Table],
) -> None:
    for name, table in tables.items():
        connection.register(name, table)


def install_diagnostics(
    connection: duckdb.DuckDBPyConnection,
    sql_path: Path,
) -> None:
    connection.execute(sql_path.read_text(encoding="utf-8"))
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.views"
        ).fetchall()
    }
    missing = set(DIAGNOSTIC_VIEWS) - existing
    if missing:
        raise ValueError(f"diagnostic SQL did not create views: {sorted(missing)}")


def request_case(
    connection: duckdb.DuckDBPyConnection,
    request_id: int,
) -> pa.Table:
    return connection.execute(
        "SELECT * FROM v4_request_case WHERE request_id = ? "
        "ORDER BY recall_rank",
        [request_id],
    ).fetch_arrow_table()
