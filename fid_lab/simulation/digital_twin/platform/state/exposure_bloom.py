"""GPU rolling Bloom filter for bounded-latency Feed exposure dedup."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import torch


@dataclass(frozen=True)
class ExposureBloomConfig:
    segments: int = 30
    bits_per_segment: int = 4_096
    hashes: int = 5
    segment_ticks: int = 96

    def __post_init__(self) -> None:
        if min(
            self.segments,
            self.bits_per_segment,
            self.hashes,
            self.segment_ticks,
        ) <= 0:
            raise ValueError("exposure Bloom dimensions must be positive")
        if self.bits_per_segment % 8:
            raise ValueError("exposure Bloom bits must be byte aligned")

    @property
    def bytes_per_segment(self) -> int:
        return self.bits_per_segment // 8

    def union_false_positive_rate(self, exposures_per_segment: int) -> float:
        one = (
            1.0
            - exp(
                -self.hashes
                * exposures_per_segment
                / self.bits_per_segment
            )
        ) ** self.hashes
        return 1.0 - (1.0 - one) ** self.segments


def build_exposure_bloom(
    users: int,
    config: ExposureBloomConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.zeros(
        users,
        config.segments,
        config.bytes_per_segment,
        dtype=torch.uint8,
        device=device,
    )
    epoch = torch.full(
        (config.segments,), -1, dtype=torch.long, device=device,
    )
    return bits, epoch


def _locations(
    item: torch.Tensor,
    config: ExposureBloomConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = item.long().clamp_min(0)
    positions = []
    for index in range(config.hashes):
        multiplier = 1_000_003 + index * 97_409
        offset = 97_531 + index * 1_299_709
        mixed = torch.bitwise_xor(
            values * multiplier + offset,
            torch.bitwise_right_shift(values, 13 + index),
        )
        positions.append(torch.remainder(mixed, config.bits_per_segment))
    position = torch.stack(positions, dim=-1)
    return torch.div(position, 8, rounding_mode="floor"), position % 8


def _rotate(
    bits: torch.Tensor,
    epochs: torch.Tensor,
    epoch: int,
    config: ExposureBloomConfig,
) -> int:
    slot = epoch % config.segments
    if int(epochs[slot]) != epoch:
        bits[:, slot].zero_()
        epochs[slot] = epoch
    return slot


def add_exposures(
    bits: torch.Tensor,
    epochs: torch.Tensor,
    user: torch.Tensor,
    item: torch.Tensor,
    event_time: torch.Tensor,
    config: ExposureBloomConfig,
) -> None:
    if not len(user):
        return
    exposure_epoch = torch.div(
        event_time, config.segment_ticks, rounding_mode="floor",
    )
    for epoch_tensor in torch.unique(exposure_epoch, sorted=True):
        epoch = int(epoch_tensor)
        selected = exposure_epoch == epoch
        slot = _rotate(bits, epochs, epoch, config)
        byte, bit = _locations(item[selected], config)
        selected_user = user[selected, None].expand_as(byte)
        byte_key = (
            (selected_user * config.segments + slot)
            * config.bytes_per_segment
            + byte
        )
        encoded = torch.unique(byte_key * 8 + bit, sorted=True)
        byte_key = torch.div(encoded, 8, rounding_mode="floor")
        bit = encoded % 8
        unique_byte, inverse = torch.unique_consecutive(
            byte_key, return_inverse=True,
        )
        mask = torch.zeros(
            len(unique_byte), dtype=torch.int16, device=bits.device,
        )
        mask.index_add_(0, inverse, (1 << bit).to(torch.int16))
        flat = bits.reshape(-1)
        flat[unique_byte] = torch.bitwise_or(
            flat[unique_byte], mask.to(torch.uint8),
        )


def contains_exposure(
    bits: torch.Tensor,
    epochs: torch.Tensor,
    user: torch.Tensor,
    item: torch.Tensor,
    logical_time: torch.Tensor,
    config: ExposureBloomConfig,
    *,
    window_ticks: int | None = None,
) -> torch.Tensor:
    if not len(user):
        return torch.zeros_like(item, dtype=torch.bool)
    current_epoch = torch.div(
        logical_time, config.segment_ticks, rounding_mode="floor",
    )
    if torch.unique(current_epoch).numel() != 1:
        raise ValueError("one Bloom query batch must share an event epoch")
    epoch = int(current_epoch[0])
    retained_segments = config.segments
    if window_ticks is not None:
        retained_segments = min(
            config.segments,
            max(
                (window_ticks + config.segment_ticks - 1)
                // config.segment_ticks,
                1,
            ),
        )
    active = (epochs <= epoch) & (epochs > epoch - retained_segments)
    union = torch.zeros(
        len(user), config.bytes_per_segment,
        dtype=torch.uint8, device=bits.device,
    )
    for slot in torch.where(active)[0].tolist():
        union.bitwise_or_(bits[user, slot])
    flattened = item.reshape(len(user), -1)
    byte, bit = _locations(flattened, config)
    present = torch.ones_like(flattened, dtype=torch.bool)
    for index in range(config.hashes):
        value = torch.gather(union, 1, byte[:, :, index])
        present &= (
            torch.bitwise_and(value, 1 << bit[:, :, index]) != 0
        )
    return (present & (flattened >= 0)).reshape_as(item)
