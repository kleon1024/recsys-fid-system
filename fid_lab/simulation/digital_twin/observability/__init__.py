"""Durable v4 full-flow observability surface."""

from .contracts import CheckpointRecord, FullFlowSnapshot
from .dataset import (
    DATASET_SCHEMA_VERSION,
    FullFlowPartitionRef,
    append_full_flow_partition,
    list_full_flow_partitions,
    open_full_flow_dataset,
    read_full_flow_partition_table,
    verify_full_flow_dataset,
    verify_full_flow_partition,
)
from .fixture import (
    FullFlowFixtureConfig,
    build_full_flow_fixture,
    build_full_flow_fixtures,
)
from .failure_fixture import seed_diagnostic_failures
from .store import (
    FULL_FLOW_SCHEMA_VERSION,
    materialize_full_flow,
    replace_json_atomic,
)
from .tables import TABLE_NAMES, build_full_flow_tables, iter_full_flow_tables

__all__ = (
    "CheckpointRecord",
    "DATASET_SCHEMA_VERSION",
    "FULL_FLOW_SCHEMA_VERSION",
    "FullFlowSnapshot",
    "FullFlowPartitionRef",
    "FullFlowFixtureConfig",
    "TABLE_NAMES",
    "build_full_flow_tables",
    "build_full_flow_fixture",
    "build_full_flow_fixtures",
    "append_full_flow_partition",
    "iter_full_flow_tables",
    "materialize_full_flow",
    "list_full_flow_partitions",
    "open_full_flow_dataset",
    "read_full_flow_partition_table",
    "replace_json_atomic",
    "seed_diagnostic_failures",
    "verify_full_flow_dataset",
    "verify_full_flow_partition",
)
