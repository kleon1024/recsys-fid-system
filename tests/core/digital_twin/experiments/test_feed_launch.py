from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from fid_lab.feed_loop.world_model.contracts import WorldModelConfig
from fid_lab.feed_loop.world_model.ensemble import WorldModelEnsemble
from fid_lab.feed_loop.world_model.feature_contract import V4_REQUIRED_FEATURES
from fid_lab.feed_loop.world_model.training import save_world_ensemble
from fid_lab.feed_loop.world_model.validation.support import SUPPORT_PROFILE_SCHEMA
from fid_lab.simulation.digital_twin import RuntimePaths, STANDARD_FEED_PROFILE
from fid_lab.simulation.digital_twin.checkpoint import (
    WorldBranchRegistry,
    WorldCheckpointStore,
)
from fid_lab.simulation.digital_twin.experiments.launch import (
    FeedLaunchSpec,
    canonical_random_policy,
    initialize_canonical_runtime,
    run_feed_launch,
)
from fid_lab.simulation.digital_twin.runtime_paths import RESPONSE_AUTHORITY_SCHEMA
from fid_lab.simulation.digital_twin.world.neural_features import (
    V4_FEATURE_CONTRACT,
    V4_FEATURE_COVERAGE,
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


def _publish_test_response_authority(paths):
    indices = sorted(V4_REQUIRED_FEATURES)
    support = {
        "schema": SUPPORT_PROFILE_SCHEMA,
        "feature_indices": indices,
        "combination": "union_of_source_components",
        "components": [{
            "name": "feed-launch-test",
            "feature_indices": indices,
            "distance_feature_indices": indices,
            "bounded_feature_indices": [],
            "bounded_lower": [],
            "bounded_upper": [],
            "center": [0.0] * len(indices),
            "scale": [1.0] * len(indices),
            "request_distance_threshold": 100.0,
        }],
    }
    ensemble = WorldModelEnsemble(WorldModelConfig(
        width=32, latent_dim=8, attention_heads=4,
        ensemble_members=2, batch_size=32, epochs=1,
    ))
    save_world_ensemble(
        ensemble,
        [[], []],
        paths.response_authority,
        {
            "manifest_sha256": "test-dataset",
            "feature_contract_sha256": V4_FEATURE_CONTRACT["sha256"],
            "feature_coverage": V4_FEATURE_COVERAGE,
        },
        support_profile=support,
    )
    manifest_path = paths.response_authority / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["authority_status"] = "accepted_feed_authority"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.response_authority_file.write_text(json.dumps({
        "schema": RESPONSE_AUTHORITY_SCHEMA,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "member_index": 0,
        "inference_batch_size": 64,
        "maximum_support_fallback_rate": 0.03,
    }, indent=2) + "\n")


def test_canonical_baseline_randomizes_order_after_random_retrieval():
    policy = canonical_random_policy(_profile())

    assert policy.enabled_routes == ("random",)
    assert policy.enabled_business_routes == ()
    assert policy.exploration_rate == 1.0


def test_canonical_runtime_rejects_missing_response_authority(tmp_path):
    profile = _profile()
    paths = RuntimePaths.standard(profile, tmp_path)
    with pytest.raises(ValueError, match="published response authority"):
        initialize_canonical_runtime(paths, profile, device="cpu")


def test_feed_launch_advances_one_factual_world_and_keeps_policy_on_aa(tmp_path):
    profile = _profile()
    paths = RuntimePaths.standard(profile, tmp_path)
    _publish_test_response_authority(paths)
    initial = initialize_canonical_runtime(paths, profile, device="cpu")
    initialization = json.loads(
        (paths.checkpoints / "refs" / f"{initial}.json").read_text()
    )
    burn_in = initialization["learning_cursors"]["world_burn_in"]
    assert initialization["logical_time"] == 7 * profile.ticks_per_day - 1
    assert burn_in["registrations_by_tick"][0] == 0
    assert burn_in["measurement_eligible_from"] == 7 * profile.ticks_per_day
    assert sum(burn_in["requests_by_tick"][-profile.ticks_per_day:]) > 0
    assert initialization["event_manifest"]["hot_batches"] < (
        initialization["event_manifest"]["batches"]
    )

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
    _publish_test_response_authority(paths)
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
    _publish_test_response_authority(paths)
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
