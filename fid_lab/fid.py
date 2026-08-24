"""Public Monolith-compatible FID bit layouts and deterministic signatures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib

import torch


class FidVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


@dataclass(frozen=True)
class FidLayout:
    slot_bits: int
    signature_bits: int

    @property
    def max_slot(self) -> int:
        return (1 << self.slot_bits) - 1

    @property
    def signature_mask(self) -> int:
        return (1 << self.signature_bits) - 1


LAYOUTS = {
    FidVersion.V1: FidLayout(slot_bits=10, signature_bits=54),
    # Bit 63 is reserved, so V2 has 15 slot bits rather than 16.
    FidVersion.V2: FidLayout(slot_bits=15, signature_bits=48),
}


class FidCodec:
    """Owns FID packing; model code must never duplicate these bit operations."""

    def __init__(self, version: FidVersion | str = FidVersion.V2) -> None:
        self.version = FidVersion(version)
        self.layout = LAYOUTS[self.version]

    def pack(self, slot: int, signature: int) -> int:
        if not 0 <= slot <= self.layout.max_slot:
            raise ValueError(
                f"slot must be in [0, {self.layout.max_slot}], got {slot}"
            )
        if signature < 0:
            raise ValueError(f"signature must be non-negative, got {signature}")
        return (slot << self.layout.signature_bits) | (
            signature & self.layout.signature_mask
        )

    def unpack(self, fid: int) -> tuple[int, int]:
        if fid < 0 or fid >= (1 << 64):
            raise ValueError("FID must be an unsigned 64-bit integer")
        if self.version is FidVersion.V2 and fid >> 63:
            raise ValueError("FID V2 reserved bit must be zero")
        slot = fid >> self.layout.signature_bits
        signature = fid & self.layout.signature_mask
        return slot, signature

    def signature(self, namespace: str, raw_value: object) -> int:
        """Return a process-stable signature; never use randomized Python hash()."""
        payload = f"{namespace}\x1f{raw_value}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8, person=b"fid-lab-v1").digest()
        return int.from_bytes(digest, "big") & self.layout.signature_mask

    def encode(self, slot: int, namespace: str, raw_value: object) -> int:
        return self.pack(slot, self.signature(namespace, raw_value))

    def encode_numeric_tensor(
        self,
        slot: int,
        namespace: str,
        raw_values: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized stable FIDs for integer-valued online tensors."""
        if not 0 <= slot <= self.layout.max_slot:
            raise ValueError(
                f"slot must be in [0, {self.layout.max_slot}], got {slot}"
            )
        if raw_values.dtype == torch.bool or raw_values.is_floating_point():
            raise TypeError("numeric tensor FIDs require an integer tensor")
        salt = self.signature(f"tensor:{namespace}", slot)
        mask = self.layout.signature_mask
        values = torch.bitwise_xor(raw_values.long(), salt) & mask
        values = torch.bitwise_xor(values, values >> 16)
        values = (values * 0x045D9F3B) & mask
        values = torch.bitwise_xor(values, values >> 16)
        values = (values * 0x045D9F3B) & mask
        signature = torch.bitwise_xor(values, values >> 16) & mask
        return (slot << self.layout.signature_bits) | signature

    @staticmethod
    def convert_v1_to_v2(fid_v1: int) -> int:
        """Match Monolith conversion; upper six V1 signature bits are discarded."""
        v1 = FidCodec(FidVersion.V1)
        slot, signature = v1.unpack(fid_v1)
        return FidCodec(FidVersion.V2).pack(slot, signature)
