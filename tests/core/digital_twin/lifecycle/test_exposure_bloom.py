from __future__ import annotations

import torch

from fid_lab.simulation.digital_twin.platform.state.exposure_bloom import (
    ExposureBloomConfig,
    add_exposures,
    build_exposure_bloom,
    contains_exposure,
)


def test_rolling_bloom_has_no_false_negatives_and_expires_segments():
    config = ExposureBloomConfig()
    bits, epochs = build_exposure_bloom(4, config, torch.device("cpu"))
    item = torch.arange(128)
    user = torch.zeros_like(item)
    event_time = torch.zeros_like(item)
    add_exposures(bits, epochs, user, item, event_time, config)

    query_user = torch.tensor([0])
    query_time = torch.tensor([0])
    assert contains_exposure(
        bits, epochs, query_user, item[None], query_time, config,
    ).all()
    expired_time = torch.tensor([config.segments * config.segment_ticks])
    assert not contains_exposure(
        bits, epochs, query_user, item[None], expired_time, config,
    ).any()


def test_rolling_bloom_false_positive_rate_stays_below_gate():
    config = ExposureBloomConfig()
    users = 32
    bits, epochs = build_exposure_bloom(users, config, torch.device("cpu"))
    user = torch.arange(users).repeat_interleave(128)
    item = torch.arange(128).repeat(users)
    add_exposures(bits, epochs, user, item, torch.zeros_like(item), config)
    absent = torch.arange(10_000, 11_000)[None].expand(users, -1)
    observed = contains_exposure(
        bits,
        epochs,
        torch.arange(users),
        absent,
        torch.zeros(users, dtype=torch.long),
        config,
    ).float().mean()

    assert float(observed) < 0.01
    assert config.union_false_positive_rate(128) < 0.01
