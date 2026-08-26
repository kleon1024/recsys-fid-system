"""Build a frozen mature-label dataset from factual request partitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from ..checkpoint import WorldBranchRegistry, WorldCheckpointStore
from ..experiments.retrieval_ladder import RetrievalLadderConfig, _build_kernel
from ..observability import (
    CheckpointRecord,
    FullFlowSnapshot,
    append_full_flow_partition,
    verify_full_flow_dataset,
)
from ..samples.joiner import JoinerConfig, RequestLevelJoiner
from ..samples.event_closure import select_joiner_events
from ..samples.publish_queue import PublishQueueConfig
from ..profile import STANDARD_FEED_PROFILE
from .request_stream import FactualRequestStream


@dataclass(frozen=True)
class RequestMaterializationConfig:
    checkpoint_root: str
    request_stream_root: str
    dataset_root: str
    checkpoint_branch: str = "main"
    users: int = STANDARD_FEED_PROFILE.users
    items: int = STANDARD_FEED_PROFILE.items
    device: str = "cuda"
    seed: int = STANDARD_FEED_PROFILE.seed
    ticks_per_day: int = STANDARD_FEED_PROFILE.ticks_per_day
    recall_negatives: int = 20
    allow_code_migration: bool = False
    allow_additive_runtime_migration: bool = False

    def __post_init__(self) -> None:
        if min(
            self.users, self.items, self.ticks_per_day, self.recall_negatives,
        ) <= 0:
            raise ValueError("request materialization dimensions must be positive")


def materialize_factual_requests(
    config: RequestMaterializationConfig,
) -> dict[str, object]:
    runtime = RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
    )
    _, kernel = _build_kernel(runtime)
    checkpoint_store = WorldCheckpointStore(Path(config.checkpoint_root))
    branch = WorldBranchRegistry(checkpoint_store).get(config.checkpoint_branch)
    if not branch.training_authority:
        raise ValueError("only factual requests can enter the training dataset")
    restored = checkpoint_store.restore(
        kernel,
        branch.head_checkpoint_id,
        require_code_match=not config.allow_code_migration,
        allow_additive_runtime_migration=config.allow_additive_runtime_migration,
    )
    stream = FactualRequestStream(
        Path(config.request_stream_root) / branch.name,
        branch,
    )
    refs = stream.refs(training=True)
    if not refs:
        raise ValueError("factual request stream is empty")
    expected_world_hash = FactualRequestStream._world_manifest_hash(
        kernel.world.manifest(),
    )
    if {ref.world_manifest_sha256 for ref in refs} != {expected_world_hash}:
        raise ValueError("request stream mixes or mismatches DGP authorities")
    event_watermark = kernel.event_log.ingest_watermark
    all_events = kernel.event_log.read(ingested_through=event_watermark)
    joiner = RequestLevelJoiner(
        JoinerConfig(
            ticks_per_day=config.ticks_per_day,
            recall_negatives=config.recall_negatives,
        ),
        kernel.world.catalog,
    )
    publish_window_ticks = max(
        task.window_ticks
        for task in PublishQueueConfig(config.ticks_per_day).tasks
    )
    output = Path(config.dataset_root)
    partitions = []
    for ref in refs:
        request = stream.read(ref, device=kernel.world.catalog.item_id.device)
        join_events = select_joiner_events(
            all_events,
            request_id=request.trace.request_id,
            user_id=request.trace.user_id,
            request_time=request.trace.event_time,
            publish_window_ticks=publish_window_ticks,
        )
        samples = joiner.materialize(
            request.trace,
            request.context,
            join_events,
            event_watermark=event_watermark,
        )
        persist = torch.isin(all_events.request_id, request.trace.request_id)
        events = all_events.select(persist)
        snapshot = FullFlowSnapshot(
            catalog=kernel.world.catalog,
            trace=request.trace,
            context=request.context,
            events=events,
            samples=samples,
            projection=kernel.platform.projection.snapshot(),
            feature_manifest=kernel.platform.ranker.features.manifest,
            checkpoints=(CheckpointRecord(
                created_time=restored.ref.logical_time,
                lane="candidate",
                model_name="factual-request-materializer",
                checkpoint_version=restored.ref.checkpoint_id,
                data_watermark=event_watermark,
                sample_manifest=stream.stream_sha256,
                feature_version=request.trace.manifest.feature_version,
                fid_version=request.trace.manifest.fid_version,
                index_version=request.trace.manifest.index_version,
                validation_status="pass",
                publish_state="training_input",
            ),),
            layer_assignment=request.layer_assignment,
        )
        partitions.append(append_full_flow_partition(
            snapshot,
            output,
            f"event_time={ref.logical_time}",
        ))
    dataset = verify_full_flow_dataset(output)
    return {
        "schema": "factual-request-materialization-review/v1",
        "quality_claim": "synthetic-world training-data lineage only",
        "config": asdict(config),
        "branch": branch.name,
        "checkpoint_id": branch.head_checkpoint_id,
        "request_stream_sha256": stream.stream_sha256,
        "event_watermark": event_watermark,
        "partitions": len(partitions),
        "dataset_content_sha256": dataset["dataset_content_sha256"],
        "table_rows": dataset["table_rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--request-stream-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-branch", default="main")
    parser.add_argument("--users", type=int, default=STANDARD_FEED_PROFILE.users)
    parser.add_argument("--items", type=int, default=STANDARD_FEED_PROFILE.items)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=STANDARD_FEED_PROFILE.seed)
    parser.add_argument(
        "--ticks-per-day", type=int,
        default=STANDARD_FEED_PROFILE.ticks_per_day,
    )
    parser.add_argument("--recall-negatives", type=int, default=20)
    parser.add_argument("--allow-code-migration", action="store_true")
    parser.add_argument("--allow-additive-runtime-migration", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = materialize_factual_requests(RequestMaterializationConfig(
        checkpoint_root=args.checkpoint_root,
        request_stream_root=args.request_stream_root,
        dataset_root=args.dataset_root,
        checkpoint_branch=args.checkpoint_branch,
        users=args.users,
        items=args.items,
        device=args.device,
        seed=args.seed,
        ticks_per_day=args.ticks_per_day,
        recall_negatives=args.recall_negatives,
        allow_code_migration=args.allow_code_migration,
        allow_additive_runtime_migration=args.allow_additive_runtime_migration,
    ))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
