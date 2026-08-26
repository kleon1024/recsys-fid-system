from __future__ import annotations

from pathlib import Path

from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.experiments.factual_collection import (
    FactualCollectionConfig,
    collect_factual_requests,
)
from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    run_retrieval_ladder,
)
from fid_lab.simulation.digital_twin.learning import (
    FactualRequestStream,
    Lane,
    PartitionedSampleBus,
)
from fid_lab.simulation.digital_twin.learning.request_materialization import (
    RequestMaterializationConfig,
    materialize_factual_requests,
)
from fid_lab.simulation.digital_twin.learning.retrieval.data import (
    load_retrieval_batch,
)
from fid_lab.simulation.digital_twin.learning.retrieval.factual_benchmark import (
    FactualRetrievalBenchmarkConfig,
    run_factual_retrieval_benchmark,
)


def test_factual_collection_resumes_main_and_persists_training_requests(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    requests = tmp_path / "requests"
    baseline = run_retrieval_ladder(RetrievalLadderConfig(
        users=128,
        items=1_200,
        burn_in_steps=1,
        experiment_steps=1,
        control_fraction=0.4,
        treatment_fraction=0.4,
        device="cpu",
        auto_promote=False,
        minimum_triggered_users=10_000,
        checkpoint_root=str(checkpoints),
        max_reviews=1,
    ))
    config = FactualCollectionConfig(
        checkpoint_root=str(checkpoints),
        request_stream_root=str(requests),
        steps=2,
        users=128,
        items=1_200,
        device="cpu",
    )
    first = collect_factual_requests(config)
    second = collect_factual_requests(config)

    assert first["resumed_from_checkpoint"] == baseline["final_checkpoint_id"]
    assert second["resumed_from_checkpoint"] == first["final_checkpoint_id"]
    assert first["logical_time"][1] + 1 == second["logical_time"][0]
    assert second["request_partitions"] == 4
    assert first["requests"] > 0 and second["requests"] > 0
    store = WorldCheckpointStore(checkpoints)
    branch = WorldBranchRegistry(store).get("main")
    assert branch.head_checkpoint_id == second["final_checkpoint_id"]
    stream = FactualRequestStream(Path(requests) / "main", branch)
    assert len(stream.refs(training=True)) == 4
    restored = store.restore(
        _kernel(config), second["final_checkpoint_id"],
    )
    cursor = restored.learning_cursors["factual_request_stream"]
    assert cursor["stream_sha256"] == stream.stream_sha256
    assert cursor["partitions"] == 4

    dataset = tmp_path / "training-dataset"
    materialized = materialize_factual_requests(RequestMaterializationConfig(
        checkpoint_root=str(checkpoints),
        request_stream_root=str(requests),
        dataset_root=str(dataset),
        users=128,
        items=1_200,
        device="cpu",
    ))
    assert materialized["partitions"] == 4
    bus = PartitionedSampleBus(dataset, tmp_path / "lane-state")
    batch = load_retrieval_batch(bus, bus.poll(Lane.CANDIDATE))
    assert len(batch.request_id) > 0
    assert batch.event_watermark == second["logical_time"][1]

    shadow = run_factual_retrieval_benchmark(
        FactualRetrievalBenchmarkConfig(
            checkpoint_root=str(checkpoints),
            dataset_root=str(dataset),
            output=str(tmp_path / "retrieval-shadow"),
            users=128,
            items=1_200,
            device="cpu",
            evaluation_partitions=1,
            epochs=1,
            batch_size=64,
            top_k=10,
            downstream_k=5,
            max_evaluation_queries=100,
        )
    )
    assert shadow["world_checkpoint_id"] == second["final_checkpoint_id"]
    assert shadow["train_partitions"] == [
        f"event_time={ref.logical_time}"
        for ref in stream.refs(training=True)[:-1]
    ]
    assert set(shadow["models"]) == {"two_tower", "multi_interest"}
    assert all(
        model["training"]["rows"] > 0 for model in shadow["models"].values()
    )


def _kernel(config: FactualCollectionConfig):
    from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
        _build_kernel,
    )

    _, kernel = _build_kernel(RetrievalLadderConfig(
        users=config.users,
        items=config.items,
        device=config.device,
        seed=config.seed,
        ticks_per_day=config.ticks_per_day,
    ))
    return kernel
