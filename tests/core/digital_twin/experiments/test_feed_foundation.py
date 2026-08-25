from __future__ import annotations

from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.experiments.feed_foundation import (
    FeedDedupLaunchConfig,
    run_feed_dedup_launch,
)
from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
    run_retrieval_ladder,
)


def test_feed_dedup_launch_backfills_ledger_and_advances_factual_head(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    baseline = run_retrieval_ladder(RetrievalLadderConfig(
        users=256,
        items=2_400,
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
    report = run_feed_dedup_launch(FeedDedupLaunchConfig(
        checkpoint_root=str(checkpoints),
        request_stream_root=str(tmp_path / "requests"),
        users=256,
        items=2_400,
        device="cpu",
        experiment_steps=1,
        control_fraction=0.4,
        treatment_fraction=0.4,
        minimum_triggered_users=10_000,
        maximum_attempts=1,
    ))
    review = report["review"]
    assert report["resumed_from_checkpoint"] == baseline["final_checkpoint_id"]
    assert review["decision"] == "stop_inconclusive"
    assert review["repeat_rate"]["treatment"] <= (
        review["repeat_rate"]["control"]
    )
    store = WorldCheckpointStore(checkpoints)
    branch = WorldBranchRegistry(store).get("main")
    assert branch.head_checkpoint_id == report["final_checkpoint_id"]
    _, kernel = _build_kernel(RetrievalLadderConfig(
        users=256,
        items=2_400,
        device="cpu",
    ))
    restored = store.restore(kernel, branch.head_checkpoint_id)
    assert int(
        kernel.platform.projection.state.user_feed_exposure_cursor.sum()
    ) > 0
    assert restored.learning_cursors["feed_dedup_launch"]["completed"]
