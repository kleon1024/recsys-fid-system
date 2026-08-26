"""One-command diagnosis for an immutable Launch Review bundle."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .dataset import open_full_flow_dataset
from .diagnostics import DIAGNOSTIC_SQL_ASSET, install_diagnostics
from .store import replace_json_atomic
from .tables import TABLE_NAMES


def _rows(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    table = connection.execute(sql).fetch_arrow_table()
    return table.to_pylist()


def _catalog_audit(connection: duckdb.DuckDBPyConnection) -> dict[str, float]:
    row = _rows(connection, """
        WITH items AS (
            SELECT item_id, any_value(topic_id) AS topic_id
            FROM v4_candidate_decision_log GROUP BY item_id
        ), ordered AS (
            SELECT item_id, topic_id,
                   lag(topic_id) OVER (ORDER BY item_id) AS prior_topic
            FROM items
        )
        SELECT
            count(DISTINCT item_id) AS observed_items,
            count(DISTINCT topic_id) AS observed_topics,
            avg(CASE WHEN topic_id = prior_topic + 1 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE prior_topic IS NOT NULL) AS adjacent_increment_rate
        FROM ordered
    """)[0]
    return {
        "observed_items": int(row["observed_items"]),
        "observed_topics": int(row["observed_topics"]),
        "adjacent_increment_rate": float(row["adjacent_increment_rate"] or 0.0),
    }


def _dedup_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    return _rows(connection, """
        WITH repeated AS (
            SELECT content_kind, user_id, item_id, count(*) AS impressions
            FROM v4_event_log
            WHERE event_name = 'impression' AND surface = 0
            GROUP BY content_kind, user_id, item_id
            HAVING count(*) > 1
        )
        SELECT content_kind,
               count(*) AS repeated_user_item_pairs,
               sum(impressions - 1) AS repeated_impressions
        FROM repeated GROUP BY content_kind ORDER BY content_kind
    """)


def _open_connection(
    full_flow_dir: Path,
) -> tuple[duckdb.DuckDBPyConnection, Path]:
    connection = duckdb.connect(":memory:")
    dataset_manifest = full_flow_dir / "dataset-manifest.json"
    if dataset_manifest.is_file():
        for name, dataset in open_full_flow_dataset(full_flow_dir).items():
            connection.register(name, dataset)
        return connection, dataset_manifest
    manifest = full_flow_dir / "manifest.json"
    for name in TABLE_NAMES:
        connection.register(
            name, pq.read_table(full_flow_dir / f"{name}.parquet"),
        )
    return connection, manifest


def diagnose_full_flow(
    full_flow_dir: Path,
    *,
    review: dict[str, object] | None = None,
) -> dict[str, object]:
    connection, identity_path = _open_connection(full_flow_dir)
    install_diagnostics(connection, DIAGNOSTIC_SQL_ASSET)
    request_cells = _rows(connection, """
        SELECT experiment_cell, count(*) AS requests,
               count(DISTINCT user_id) AS users
        FROM v4_request_log GROUP BY experiment_cell ORDER BY experiment_cell
    """)
    routes = _rows(connection, """
        SELECT route.route_name,
               count(*) AS candidates,
               count(DISTINCT route.request_id) AS requests,
               count(DISTINCT route.item_id) AS unique_items,
               avg(CAST(
                   route.country = request.user_country AS DOUBLE
               )) AS country_match_rate,
               avg(per_request.topics) AS topics_per_request,
               avg(per_request.items) AS items_per_request,
               avg(route.content_age) AS mean_age_ticks,
               avg(route.duration_seconds) AS mean_duration_seconds,
               avg(route.quality_prior) AS mean_quality_prior,
               avg(route.recent_engagements /
                   greatest(route.recent_impressions, 1.0)) AS engagement_rate,
               avg(route.recent_negatives /
                   greatest(route.recent_impressions, 1.0)) AS negative_rate
        FROM v4_route_candidate_log AS route
        JOIN v4_request_log AS request USING (request_id)
        JOIN (
            SELECT request_id, route_name,
                   count(DISTINCT topic_id) AS topics,
                   count(DISTINCT item_id) AS items
            FROM v4_route_candidate_log GROUP BY request_id, route_name
        ) AS per_request USING (request_id, route_name)
        GROUP BY route.route_name ORDER BY route.route_name
    """)
    stages = _rows(connection, """
        SELECT routes.route_names,
               count(*) AS recalled,
               count(*) FILTER (WHERE candidate.coarse_pass) AS coarse,
               count(*) FILTER (WHERE candidate.fine_pass) AS fine,
               count(*) FILTER (WHERE candidate.exposed) AS exposed
        FROM v4_candidate_decision_log AS candidate
        LEFT JOIN v4_route_items AS routes USING (request_id, item_id)
        GROUP BY routes.route_names ORDER BY routes.route_names
    """)
    stage_pressure = _rows(connection, """
        WITH request_stage AS (
            SELECT request_id,
                   count(*) AS recalled,
                   count(*) FILTER (WHERE coarse_pass) AS coarse,
                   count(*) FILTER (WHERE fine_pass) AS fine,
                   count(*) FILTER (WHERE exposed) AS exposed
            FROM v4_candidate_decision_log GROUP BY request_id
        )
        SELECT avg(recalled) AS mean_recalled,
               avg(coarse) AS mean_coarse,
               avg(fine) AS mean_fine,
               avg(exposed) AS mean_exposed,
               avg(CAST(recalled = coarse AS DOUBLE)) AS coarse_noop_request_rate
        FROM request_stage
    """)[0]
    sample_support = _rows(connection, """
        SELECT count(*) AS fine_rows,
               count(*) FILTER (WHERE exposed) AS exposed_rows,
               count(*) FILTER (WHERE randomized_support) AS randomized_rows,
               count(*) FILTER (WHERE ope_supported) AS ope_rows,
               count(*) FILTER (
                   WHERE list_has_any(task_label_masks, [true])
               ) AS labeled_rows,
               count(*) FILTER (
                   WHERE randomized_support
                     AND list_has_any(task_label_masks, [true])
               ) AS randomized_labeled_rows
        FROM v4_training_example_log WHERE authority = 'fine'
    """)[0]
    exposure = _rows(connection, """
        SELECT request.experiment_cell,
               count(*) AS exposures,
               count(DISTINCT candidate.item_id) AS unique_items,
               count(DISTINCT candidate.topic_id) AS unique_topics,
               max(item_count) / count(*) AS top_item_share
        FROM v4_candidate_decision_log AS candidate
        JOIN v4_request_log AS request USING (request_id)
        JOIN (
            SELECT experiment_cell, item_id, count(*) AS item_count
            FROM v4_candidate_decision_log
            JOIN v4_request_log USING (request_id)
            WHERE exposed GROUP BY experiment_cell, item_id
        ) AS concentration USING (experiment_cell, item_id)
        WHERE candidate.exposed
        GROUP BY request.experiment_cell ORDER BY request.experiment_cell
    """)
    dedup = _dedup_rows(connection)
    catalog = _catalog_audit(connection)
    findings = []
    if catalog["adjacent_increment_rate"] > 0.20:
        findings.append({
            "severity": "blocker",
            "code": "catalog_topic_periodicity",
            "evidence": catalog["adjacent_increment_rate"],
        })
    for route in routes:
        if route["route_name"] == "popular" and route["country_match_rate"] < 0.95:
            findings.append({
                "severity": "blocker",
                "code": "popular_market_mismatch",
                "evidence": route["country_match_rate"],
            })
        if route["unique_items"] < route["items_per_request"] * 2:
            findings.append({
                "severity": "warning",
                "code": f"{route['route_name']}_pool_concentration",
                "evidence": route["unique_items"],
            })
    repeated_impressions = sum(
        int(row["repeated_impressions"]) for row in dedup
    )
    if repeated_impressions:
        findings.append({
            "severity": "blocker",
            "code": "feed_item_repeat",
            "evidence": repeated_impressions,
        })
    if float(stage_pressure["coarse_noop_request_rate"] or 0.0) > 0.95:
        findings.append({
            "severity": "warning",
            "code": "coarse_stage_has_no_candidate_pressure",
            "evidence": stage_pressure["coarse_noop_request_rate"],
        })
    if int(sample_support["randomized_labeled_rows"] or 0) == 0:
        findings.append({
            "severity": "warning",
            "code": "fine_labels_have_no_randomized_support",
            "evidence": sample_support["labeled_rows"],
        })
    diagnosis = {
        "schema": "launch-diagnosis/v1",
        "full_flow_manifest_sha256": sha256(identity_path.read_bytes()).hexdigest(),
        "review": review or {},
        "request_cells": request_cells,
        "catalog": catalog,
        "routes": routes,
        "stages": stages,
        "stage_pressure": stage_pressure,
        "sample_support": sample_support,
        "exposure": exposure,
        "dedup": dedup,
        "findings": findings,
        "diagnostic_views": list(TABLE_NAMES),
    }
    return diagnosis


def _markdown(diagnosis: dict[str, object]) -> str:
    lines = ["# Launch diagnosis", "", "## Findings", ""]
    findings = diagnosis["findings"]
    if not findings:
        lines.append("No structural blocker detected.")
    else:
        lines.extend(
            f"- {item['severity']}: `{item['code']}` = {item['evidence']}"
            for item in findings
        )
    lines.extend(("", "## Route evidence", "", "```json"))
    lines.append(json.dumps(diagnosis["routes"], indent=2))
    lines.extend(("```", ""))
    return "\n".join(lines)


def write_diagnosis(
    full_flow_dir: Path,
    output_dir: Path,
    *,
    review: dict[str, object] | None = None,
) -> dict[str, object]:
    diagnosis = diagnose_full_flow(full_flow_dir, review=review)
    output_dir.mkdir(parents=True, exist_ok=True)
    replace_json_atomic(output_dir / "diagnosis.json", diagnosis)
    (output_dir / "diagnosis.md").write_text(
        _markdown(diagnosis), encoding="utf-8",
    )
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    diagnosis = write_diagnosis(
        (
            args.bundle / "full-flow-dataset"
            if (args.bundle / "full-flow-dataset").exists()
            else args.bundle / "full-flow"
        ),
        args.bundle,
    )
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
