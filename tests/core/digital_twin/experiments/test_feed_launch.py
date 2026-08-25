from __future__ import annotations

from dataclasses import replace

import pytest

from fid_lab.simulation.digital_twin import RuntimePaths, STANDARD_FEED_PROFILE
from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.experiments.launch import (
    FeedLaunchSpec,
    initialize_canonical_runtime,
    run_feed_launch,
)


def _profile():
    return replace(
        STANDARD_FEED_PROFILE,
        name="feed-launch-test-v1",
        users=256,
        items=4_000,
        ticks_per_day=8,
        feed_exposure_history_length=256,
    )


def _aa_spec():
    return FeedLaunchSpec(
        launch_id="F-AA-00",
        kind="aa",
        hypothesis="identical policies produce a valid A/A",
        isolated_change="none",
        primary_metric="dwell_seconds",
        treatment_changes={},
        experiment_seed=8_401,
        minimum_triggered_users=2,
        minimum_ticks=2,
        maximum_ticks=4,
    )


def test_feed_launch_advances_one_factual_world_and_keeps_policy_on_aa(tmp_path):
    profile = _profile()
    paths = RuntimePaths.standard(profile, tmp_path)
    initial = initialize_canonical_runtime(paths, profile, device="cpu")

    report = run_feed_launch(
        paths,
        _aa_spec(),
        profile,
        device="cpu",
        source_revision="test-source",
    )

    branch = WorldBranchRegistry(
        WorldCheckpointStore(paths.checkpoints)
    ).get("main")
    assert branch.head_checkpoint_id != initial
    assert branch.head_checkpoint_id == report["result_checkpoint_id"]
    assert report["parent_checkpoint_id"] == initial
    assert report["feed_repeat"]["repeated_impressions"] == 0
    assert (paths.launch_journal / "F-AA-00.json").is_file()
    assert len(tuple((paths.request_stream / "main").glob("**/*.json"))) > 0


def test_feed_launch_id_cannot_be_reused_on_the_same_world(tmp_path):
    profile = _profile()
    paths = RuntimePaths.standard(profile, tmp_path)
    run_feed_launch(
        paths,
        _aa_spec(),
        profile,
        device="cpu",
        source_revision="test-source",
    )

    try:
        run_feed_launch(
            paths,
            _aa_spec(),
            profile,
            device="cpu",
            source_revision="test-source",
        )
    except ValueError as error:
        assert "already completed" in str(error)
    else:
        raise AssertionError("one factual world cannot repeat a launch id")


def test_feed_launch_fails_closed_on_uninstalled_model_version(tmp_path):
    profile = _profile()
    paths = RuntimePaths.standard(profile, tmp_path)
    spec = replace(
        _aa_spec(),
        launch_id="F-F01",
        kind="policy",
        treatment_changes={"fine_version_id": 999},
    )

    with pytest.raises(KeyError, match="unknown model version"):
        run_feed_launch(
            paths,
            spec,
            profile,
            device="cpu",
            source_revision="test-source",
        )
