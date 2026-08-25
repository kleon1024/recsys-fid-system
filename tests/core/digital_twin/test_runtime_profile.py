from __future__ import annotations

import pytest

from fid_lab.simulation.digital_twin.experiments.retrieval_ladder import (
    RetrievalLadderConfig,
    _build_kernel,
)
from fid_lab.simulation.digital_twin.profile import STANDARD_FEED_PROFILE


@pytest.mark.parametrize("ticks_per_day", (8, 96))
def test_one_resolved_clock_drives_world_platform_lifecycle_and_features(
    ticks_per_day: int,
):
    config = RetrievalLadderConfig(
        users=32,
        items=900,
        ticks_per_day=ticks_per_day,
        device="cpu",
    )
    _, kernel = _build_kernel(config)

    assert kernel.world.config.ticks_per_day == ticks_per_day
    assert kernel.platform.config.ticks_per_day == ticks_per_day
    assert kernel.platform.projection.lifecycle_config.ticks_per_day == ticks_per_day
    assert kernel.platform.ranker.features.ticks_per_day == ticks_per_day
    assert config.simulation_profile.ticks_per_day == ticks_per_day


def test_default_feed_profile_has_one_stable_scale_contract():
    config = RetrievalLadderConfig()

    assert config.simulation_profile == STANDARD_FEED_PROFILE
    assert STANDARD_FEED_PROFILE.manifest()["schema"] == "simulation-profile/v1"
    assert len(STANDARD_FEED_PROFILE.profile_hash) == 64
