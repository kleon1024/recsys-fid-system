"""Content-bound Parquet materialization for v4 analytical tables."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq

from .contracts import FullFlowSnapshot
from .failure_fixture import seed_diagnostic_failures
from .tables import iter_full_flow_tables


FULL_FLOW_SCHEMA_VERSION = "digital-twin-full-flow-v3"


def _replace_json(path: Path, value: dict[str, object]) -> None:
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def materialize_full_flow(
    snapshot: FullFlowSnapshot,
    output_dir: Path,
    *,
    row_group_size: int = 131_072,
    seed_failures: bool = False,
) -> dict[str, object]:
    """Write one immutable full-flow partition and its content manifest."""
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if seed_failures and len(snapshot.trace.request_id) > 10_000:
        raise ValueError("diagnostic failure fixture is limited to 10K requests")
    table_items = iter_full_flow_tables(snapshot)
    if seed_failures:
        table_items = iter(seed_diagnostic_failures(dict(table_items)).items())
    manifest: dict[str, object] = {
        "schema": FULL_FLOW_SCHEMA_VERSION,
        "diagnostic_failure_injection": seed_failures,
        "event_watermark": snapshot.samples.event_watermark,
        "event_time_min": int(snapshot.trace.event_time.min()),
        "event_time_max": int(snapshot.trace.event_time.max()),
        "ingest_time_min": (
            int(snapshot.events.ingest_time.min())
            if len(snapshot.events.ingest_time) else -1
        ),
        "ingest_time_max": (
            int(snapshot.events.ingest_time.max())
            if len(snapshot.events.ingest_time) else -1
        ),
        "trace_manifest": {
            "schema_version": snapshot.trace.manifest.schema_version,
            "feature_version": snapshot.trace.manifest.feature_version,
            "catalog_version": snapshot.trace.manifest.catalog_version,
            "policy_registry_version": (
                snapshot.trace.manifest.policy_registry_version
            ),
            "route_names": list(snapshot.trace.manifest.route_names),
            "index_version": snapshot.trace.manifest.index_version,
            "fid_version": snapshot.trace.manifest.fid_version,
            "lifecycle_version": snapshot.trace.manifest.lifecycle_version,
        },
        "tables": {},
    }
    for name, table in table_items:
        path = output_dir / f"{name}.parquet"
        with NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{name}.",
            suffix=".parquet.tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        try:
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                row_group_size=row_group_size,
                write_statistics=True,
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest["tables"][name] = {
            "asset_key": f"observability.{name}",
            "file": path.name,
            "rows": len(table),
            "schema": str(table.schema),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path = output_dir / "manifest.json"
    _replace_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    return manifest
