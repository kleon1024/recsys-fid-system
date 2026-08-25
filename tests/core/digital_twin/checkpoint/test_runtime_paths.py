from __future__ import annotations

from dataclasses import replace

from fid_lab.simulation.digital_twin import (
    RuntimePaths,
    STANDARD_FEED_PROFILE,
)
from fid_lab.simulation.digital_twin.experiments.launch_cli import (
    resolve_runtime_paths,
)


def test_runtime_root_binds_every_state_path_to_one_profile(tmp_path):
    paths = RuntimePaths.standard(STANDARD_FEED_PROFILE, tmp_path)

    first = paths.initialize(STANDARD_FEED_PROFILE)
    second = paths.initialize(STANDARD_FEED_PROFILE)

    assert first == second == paths.manifest()
    assert first["profile_hash"] == STANDARD_FEED_PROFILE.profile_hash
    assert all(path.is_dir() for path in paths.data_paths())


def test_runtime_root_rejects_profile_drift(tmp_path):
    paths = RuntimePaths.standard(STANDARD_FEED_PROFILE, tmp_path)
    paths.initialize(STANDARD_FEED_PROFILE)
    changed = replace(STANDARD_FEED_PROFILE, ticks_per_day=8)

    try:
        paths.initialize(changed)
    except ValueError as error:
        assert "another simulation profile" in str(error)
    else:
        raise AssertionError("one runtime root cannot mix simulation profiles")


def test_launch_cli_treats_explicit_runtime_root_as_exact(tmp_path):
    assert resolve_runtime_paths(tmp_path).root == tmp_path
