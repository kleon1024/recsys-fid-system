"""Partitioned, content-verified full-flow dataset authority."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import mkdtemp

from filelock import FileLock
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as pq

from .contracts import FullFlowSnapshot
from .store import (
    FULL_FLOW_SCHEMA_VERSION,
    materialize_full_flow,
    replace_json_atomic,
)
from .tables import TABLE_NAMES


DATASET_SCHEMA_VERSION = "digital-twin-full-flow-dataset-v3"
_PARTITION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]{0,127}$")


@dataclass(frozen=True)
class FullFlowPartitionRef:
    key: str
    content_sha256: str
    manifest_sha256: str
    event_watermark: int
    event_time_min: int
    event_time_max: int


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_content(manifest: dict[str, object]) -> str:
    identity = {
        "schema": manifest["schema"],
        "diagnostic_failure_injection": manifest[
            "diagnostic_failure_injection"
        ],
        "event_watermark": manifest["event_watermark"],
        "event_time_min": manifest["event_time_min"],
        "event_time_max": manifest["event_time_max"],
        "ingest_time_min": manifest["ingest_time_min"],
        "ingest_time_max": manifest["ingest_time_max"],
        "trace_manifest": manifest["trace_manifest"],
        "sample_contract": manifest["sample_contract"],
        "feature_manifest": manifest["feature_manifest"],
        "tables": {
            name: {
                "rows": table["rows"],
                "sha256": table["sha256"],
            }
            for name, table in sorted(manifest["tables"].items())
        },
    }
    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()


def _dataset_contract(manifest: dict[str, object]) -> dict[str, object]:
    trace = manifest["trace_manifest"]
    return {
        "sample_contract": manifest["sample_contract"],
        "feature_manifest": manifest["feature_manifest"],
        "trace_manifest": {
            key: trace[key]
            for key in (
                "schema_version",
                "feature_version",
                "catalog_version",
                "policy_registry_version",
                "route_names",
                "fid_version",
                "lifecycle_version",
                "feature_manifest_hash",
            )
        },
    }


def verify_full_flow_partition(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"partition manifest is missing: {path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != FULL_FLOW_SCHEMA_VERSION:
        raise ValueError("partition schema is unsupported")
    for name, evidence in manifest["tables"].items():
        table_path = path / evidence["file"]
        if not table_path.is_file():
            raise ValueError(f"partition table is missing: {name}")
        digest = sha256(table_path.read_bytes()).hexdigest()
        if digest != evidence["sha256"]:
            raise ValueError(f"partition table hash mismatch: {name}")
        metadata = pq.read_metadata(table_path)
        if metadata.num_rows != evidence["rows"]:
            raise ValueError(f"partition table row count mismatch: {name}")
    content = _partition_content(manifest)
    recorded = manifest.get("partition_content_sha256")
    if recorded is not None and recorded != content:
        raise ValueError("partition content identity mismatch")
    manifest["partition_content_sha256"] = content
    manifest["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    return manifest


def _write_partition_manifest(
    directory: Path,
    partition_key: str,
) -> dict[str, object]:
    path = directory / "manifest.json"
    manifest = _read_json(path)
    event_partition = re.fullmatch(r"event_time=(-?\d+)", partition_key)
    if event_partition is not None:
        expected = int(event_partition.group(1))
        if not (
            manifest["event_time_min"] == expected
            and manifest["event_time_max"] == expected
        ):
            raise ValueError("event-time partition key does not match data")
    manifest["partition_key"] = partition_key
    manifest["partition_content_sha256"] = _partition_content(manifest)
    replace_json_atomic(path, manifest)
    return verify_full_flow_partition(directory)


def _new_dataset_manifest(contract: dict[str, object]) -> dict[str, object]:
    return {
        "schema": DATASET_SCHEMA_VERSION,
        "contract": contract,
        "partitions": {},
        "table_rows": {},
        "table_schemas": {},
    }


def _recompute_dataset(manifest: dict[str, object]) -> None:
    totals: dict[str, int] = {}
    for partition in manifest["partitions"].values():
        for name, table in partition["tables"].items():
            totals[name] = totals.get(name, 0) + int(table["rows"])
    manifest["table_rows"] = dict(sorted(totals.items()))
    payload = json.dumps(
        {
            "schema": manifest["schema"],
            "contract": manifest["contract"],
            "partitions": manifest["partitions"],
            "table_rows": manifest["table_rows"],
            "table_schemas": manifest["table_schemas"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest["dataset_content_sha256"] = sha256(payload).hexdigest()


def verify_full_flow_dataset(root: Path) -> dict[str, object]:
    path = root / "dataset-manifest.json"
    if not path.is_file():
        raise ValueError("dataset manifest is missing")
    manifest = _read_json(path)
    if manifest.get("schema") != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset schema is unsupported")
    for key, recorded in manifest["partitions"].items():
        actual = verify_full_flow_partition(root / "partitions" / key)
        if actual["partition_content_sha256"] != recorded["content_sha256"]:
            raise ValueError(f"dataset partition identity mismatch: {key}")
        if actual["manifest_sha256"] != recorded["manifest_sha256"]:
            raise ValueError(f"dataset partition manifest mismatch: {key}")
        actual_schemas = {
            name: table["schema"] for name, table in actual["tables"].items()
        }
        if actual_schemas != manifest["table_schemas"]:
            raise ValueError(f"dataset partition schema mismatch: {key}")
    expected = manifest.get("dataset_content_sha256")
    _recompute_dataset(manifest)
    if expected != manifest["dataset_content_sha256"]:
        raise ValueError("dataset content identity mismatch")
    return manifest


def open_full_flow_dataset(
    root: Path,
) -> dict[str, arrow_dataset.Dataset]:
    """Open verified lazy Arrow datasets, one per analytical authority."""
    manifest = verify_full_flow_dataset(root)
    partitions = tuple(manifest["partitions"])
    if not partitions:
        raise ValueError("full-flow dataset has no partitions")
    return {
        name: arrow_dataset.dataset(
            [
                root / "partitions" / key / f"{name}.parquet"
                for key in partitions
            ],
            format="parquet",
        )
        for name in TABLE_NAMES
    }


def list_full_flow_partitions(root: Path) -> tuple[FullFlowPartitionRef, ...]:
    """Resolve verified event-time partitions without exposing path conventions."""
    manifest = verify_full_flow_dataset(root)
    refs = tuple(
        FullFlowPartitionRef(
            key=key,
            content_sha256=str(partition["content_sha256"]),
            manifest_sha256=str(partition["manifest_sha256"]),
            event_watermark=int(partition["event_watermark"]),
            event_time_min=int(partition["event_time_min"]),
            event_time_max=int(partition["event_time_max"]),
        )
        for key, partition in manifest["partitions"].items()
    )
    return tuple(sorted(refs, key=lambda ref: (ref.event_watermark, ref.key)))


def read_full_flow_partition_table(
    root: Path,
    ref: FullFlowPartitionRef,
    table_name: str,
    *,
    columns: tuple[str, ...] | None = None,
):
    """Read one verified logical table from one content-bound partition."""
    if table_name not in TABLE_NAMES:
        raise ValueError(f"unknown full-flow table: {table_name}")
    current = {item.key: item for item in list_full_flow_partitions(root)}
    if current.get(ref.key) != ref:
        raise ValueError("partition reference is stale or incompatible")
    return pq.read_table(
        root / "partitions" / ref.key / f"{table_name}.parquet",
        columns=list(columns) if columns is not None else None,
    )


def _install_partition(
    root: Path,
    staging: Path,
    partition_key: str,
    incoming: dict[str, object],
) -> dict[str, object]:
    manifest_path = root / "dataset-manifest.json"
    target = root / "partitions" / partition_key
    dataset = (
        verify_full_flow_dataset(root)
        if manifest_path.exists()
        else _new_dataset_manifest(_dataset_contract(incoming))
    )
    if dataset["contract"] != _dataset_contract(incoming):
        raise ValueError("partition contract differs from dataset")
    incoming_schemas = {
        name: table["schema"] for name, table in incoming["tables"].items()
    }
    if dataset["table_schemas"] and (
        dataset["table_schemas"] != incoming_schemas
    ):
        raise ValueError("partition table schemas differ from dataset")
    dataset["table_schemas"] = incoming_schemas
    if target.exists():
        existing = verify_full_flow_partition(target)
        if (
            existing["partition_content_sha256"]
            != incoming["partition_content_sha256"]
        ):
            raise ValueError("partition key already has different content")
        return {
            "status": "resumed",
            "partition_key": partition_key,
            "partition_content_sha256": existing[
                "partition_content_sha256"
            ],
            "dataset_content_sha256": dataset["dataset_content_sha256"],
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    dataset["partitions"][partition_key] = {
        "content_sha256": incoming["partition_content_sha256"],
        "manifest_sha256": incoming["manifest_sha256"],
        "event_watermark": incoming["event_watermark"],
        "event_time_min": incoming["event_time_min"],
        "event_time_max": incoming["event_time_max"],
        "ingest_time_min": incoming["ingest_time_min"],
        "ingest_time_max": incoming["ingest_time_max"],
                "tables": {
                    name: {
                        "rows": table["rows"],
                        "sha256": table["sha256"],
                        "schema": table["schema"],
                    }
            for name, table in incoming["tables"].items()
        },
    }
    dataset["partitions"] = dict(sorted(dataset["partitions"].items()))
    _recompute_dataset(dataset)
    replace_json_atomic(manifest_path, dataset)
    verified = verify_full_flow_dataset(root)
    return {
        "status": "written",
        "partition_key": partition_key,
        "partition_content_sha256": incoming["partition_content_sha256"],
        "dataset_content_sha256": verified["dataset_content_sha256"],
    }


def append_full_flow_partition(
    snapshot: FullFlowSnapshot,
    root: Path,
    partition_key: str,
    *,
    seed_failures: bool = False,
) -> dict[str, object]:
    """Atomically append or exactly resume one event-time partition."""
    if not _PARTITION.fullmatch(partition_key):
        raise ValueError("partition key contains unsupported characters")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=".full-flow-", dir=root))
    try:
        materialize_full_flow(
            snapshot,
            staging,
            seed_failures=seed_failures,
        )
        incoming = _write_partition_manifest(staging, partition_key)
        with FileLock(str(root / ".dataset.lock")):
            return _install_partition(
                root, staging, partition_key, incoming,
            )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
