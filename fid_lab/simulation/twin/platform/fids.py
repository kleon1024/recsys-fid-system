"""Versioned sparse FID authority shared by training and online serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ....fid import FidCodec, FidVersion


TWIN_FID_SCHEMA_VERSION = "twin-sparse-fids-v1"


@dataclass(frozen=True)
class TensorFidField:
    name: str
    slot: int
    buckets: int


TWIN_FID_FIELDS = (
    TensorFidField("user_id", 1, 1_048_576),
    TensorFidField("item_id", 101, 2_097_152),
    TensorFidField("item_kind", 102, 32),
    TensorFidField("surface", 201, 16),
    TensorFidField("route", 202, 16),
    TensorFidField("hour", 204, 32),
    TensorFidField("user_x_kind", 301, 262_144),
    TensorFidField("surface_x_kind", 302, 256),
)


class TwinFidEncoder:
    version = TWIN_FID_SCHEMA_VERSION

    def __init__(self):
        self.codec = FidCodec(FidVersion.V2)

    def encode_candidates(
        self,
        *,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        item_kind: torch.Tensor,
        surface: torch.Tensor,
        route: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = item_id.shape
        user = user_id.long()[:, None].expand(shape)
        item = item_id.long()
        kind = item_kind.long()
        request_surface = surface.long()[:, None].expand(shape)
        request_step = step.long()[:, None].expand(shape)
        candidate_route = route.long()
        raw = {
            "user_id": user,
            "item_id": item,
            "item_kind": kind,
            "surface": request_surface,
            "route": candidate_route,
            "hour": torch.remainder(request_step, 24),
            "user_x_kind": user * 37 + kind,
            "surface_x_kind": request_surface * 37 + kind,
        }
        valid = item >= 0
        fids, buckets = [], []
        for field in TWIN_FID_FIELDS:
            encoded = self.codec.encode_numeric_tensor(
                field.slot, field.name, raw[field.name].clamp_min(0)
            )
            bucket = 1 + torch.remainder(encoded, field.buckets - 1)
            fids.append(encoded.masked_fill(~valid, 0))
            buckets.append(bucket.masked_fill(~valid, 0))
        return torch.stack(fids, dim=2), torch.stack(buckets, dim=2)

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "codec": self.codec.version.value,
            "missing_bucket": 0,
            "fields": [asdict(field) for field in TWIN_FID_FIELDS],
        }
