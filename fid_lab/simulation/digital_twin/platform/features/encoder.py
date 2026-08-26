"""GPU-vectorized feature compiler used by the online cascade."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .....fid import FidCodec
from ...catalog import PublicCatalog
from ...contracts import EventType, PlatformRequestBatch
from ..projection import ITEM_COUNTER_EVENTS, PlatformProjectionState
from ..retrieval import MultiRouteRetriever
from ..sequences import resolve_user_sequence
from .manifest import DEFAULT_FEATURE_MANIFEST, FeatureManifest


@dataclass(frozen=True)
class FeatureTensorBatch:
    dense: torch.Tensor
    sparse_fids: torch.Tensor
    sparse_buckets: torch.Tensor
    manifest_hash: str

    def __post_init__(self) -> None:
        if self.dense.ndim != 3 or self.sparse_fids.ndim != 3:
            raise ValueError("feature tensors must be [request, candidate, field]")
        if self.sparse_fids.shape != self.sparse_buckets.shape:
            raise ValueError("sparse FIDs and buckets must align")
        if self.dense.shape[:2] != self.sparse_fids.shape[:2]:
            raise ValueError("dense and sparse candidate axes must align")
        if not self.manifest_hash:
            raise ValueError("feature manifest hash is required")


class PlatformFeatureEncoder:
    def __init__(
        self,
        catalog: PublicCatalog,
        retriever: MultiRouteRetriever,
        ticks_per_day: int,
        manifest: FeatureManifest = DEFAULT_FEATURE_MANIFEST,
    ) -> None:
        if ticks_per_day <= 0:
            raise ValueError("ticks per day must be positive")
        self.catalog = catalog
        self.retriever = retriever
        self.ticks_per_day = ticks_per_day
        self.manifest = manifest
        self.codec = FidCodec(manifest.fid_version)

    def base_dense(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
        recall_score: torch.Tensor,
    ) -> torch.Tensor:
        item = item_id.clamp_min(0)
        valid = item_id >= 0
        query = self.retriever.query_embedding(requests, state)
        affinity = torch.einsum(
            "bkd,bd->bk", self.catalog.content_embedding[item], query,
        )
        impression = state.item_event_counts[
            item, ITEM_COUNTER_EVENTS.index(EventType.IMPRESSION)
        ]
        engagement = (
            state.item_event_counts[
                item, ITEM_COUNTER_EVENTS.index(EventType.LONG_VIEW)
            ]
            + state.item_event_counts[
                item, ITEM_COUNTER_EVENTS.index(EventType.CLICK)
            ]
        )
        age = (
            requests.event_time[:, None]
            - state.item_publish_time[item].clamp_max(requests.event_time[:, None])
        ).clamp_min(0).float()
        country = state.user_country[requests.user_id]
        region = state.user_region[requests.user_id]
        creator = state.item_creator_id[item]
        history = state.user_history_item[requests.user_id]
        features = torch.stack((
            affinity,
            self.catalog.quality_prior[item],
            torch.log1p(impression) / 8.0,
            (engagement / impression.clamp_min(1.0)).clamp_max(1.0),
            torch.exp(-age / (2.0 * self.ticks_per_day)),
            0.35 * (self.catalog.country[item] == country[:, None]).float()
            + 0.65 * (self.catalog.region[item] == region[:, None]).float(),
            state.item_inventory[item],
            recall_score.clamp_min(0.0) * 20.0,
            (
                state.creator_engagements[creator]
                / state.creator_impressions[creator].clamp_min(1.0)
            ).clamp_max(1.0),
            (item[:, :, None] == history[:, None, :]).any(dim=2).float(),
        ), dim=2)
        return features.masked_fill(~valid[:, :, None], 0.0)

    def sequence_affinity(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
    ) -> torch.Tensor:
        item = item_id.clamp_min(0)
        sequence = resolve_user_sequence(
            state, requests.user_id, requests.event_time,
        )
        history = sequence.item_id
        history_valid = sequence.strong_mask()
        candidate = self.catalog.content_embedding[item]
        history_embedding = self.catalog.content_embedding[history.clamp_min(0)]
        similarity = torch.einsum(
            "bkd,bhd->bkh", candidate, history_embedding,
        ).masked_fill(~history_valid[:, None, :], -20.0)
        attention = torch.softmax(similarity, dim=2)
        attention = attention * history_valid[:, None, :].float()
        attention /= attention.sum(dim=2, keepdim=True).clamp_min(1e-8)
        interest = torch.einsum("bkh,bhd->bkd", attention, history_embedding)
        score = torch.einsum("bkd,bkd->bk", candidate, interest)
        return score.masked_fill(item_id < 0, 0.0)

    def encode(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
        recall_score: torch.Tensor,
        route_bits: torch.Tensor,
    ) -> FeatureTensorBatch:
        sequence = self.sequence_affinity(requests, state, item_id)
        dense = torch.cat((
            self.base_dense(requests, state, item_id, recall_score),
            sequence[:, :, None],
        ), dim=2).float()
        raw = self._sparse_raw(requests, state, item_id, route_bits)
        fids, buckets = [], []
        for field, values in zip(self.manifest.sparse_fields, raw, strict=True):
            encoded = self.codec.encode_numeric_tensor(
                field.slot, field.namespace, values,
            )
            bucket = (encoded & self.codec.layout.signature_mask) % field.buckets
            valid = item_id >= 0
            fids.append(torch.where(valid, encoded, torch.zeros_like(encoded)))
            buckets.append(torch.where(valid, bucket, torch.zeros_like(bucket)))
        result = FeatureTensorBatch(
            dense=dense,
            sparse_fids=torch.stack(fids, dim=2),
            sparse_buckets=torch.stack(buckets, dim=2),
            manifest_hash=self.manifest.manifest_hash,
        )
        if result.dense.shape[2] != len(self.manifest.dense_fields):
            raise AssertionError("dense compiler and manifest diverged")
        return result

    def _sparse_raw(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
        route_bits: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        item = item_id.clamp_min(0)
        shape = item_id.shape
        user = requests.user_id[:, None].expand(shape)
        surface = requests.surface[:, None].expand(shape)
        content_kind = self.catalog.content_kind[item]
        country = state.user_country[requests.user_id][:, None].expand(shape)
        region = state.user_region[requests.user_id][:, None].expand(shape)
        hour = (
            torch.remainder(requests.event_time, self.ticks_per_day) * 24
            // self.ticks_per_day
        )[:, None].expand(shape)
        return (
            user,
            item,
            state.item_creator_id[item],
            content_kind,
            self.catalog.topic_id[item],
            surface,
            country,
            region,
            hour,
            route_bits.long(),
            state.item_lifecycle[item],
            user * 16 + content_kind,
            surface * 16 + content_kind,
        )

    def collision_report(
        self,
        requests: PlatformRequestBatch,
        state: PlatformProjectionState,
        item_id: torch.Tensor,
        route_bits: torch.Tensor,
    ) -> dict[str, object]:
        raw_fields = self._sparse_raw(requests, state, item_id, route_bits)
        valid = item_id >= 0
        fields: dict[str, object] = {}
        for field, raw in zip(
            self.manifest.sparse_fields, raw_fields, strict=True,
        ):
            values = raw[valid]
            encoded = self.codec.encode_numeric_tensor(
                field.slot, field.namespace, values,
            )
            buckets = (encoded & self.codec.layout.signature_mask) % field.buckets
            raw_unique = int(torch.unique(values).numel())
            fid_unique = int(torch.unique(encoded).numel())
            bucket_unique = int(torch.unique(buckets).numel())
            fields[field.name] = {
                "observations": int(values.numel()),
                "raw_unique": raw_unique,
                "fid_unique": fid_unique,
                "bucket_unique": bucket_unique,
                "fid_collisions": raw_unique - fid_unique,
                "bucket_collisions": raw_unique - bucket_unique,
            }
        return {
            "schema": "feature-collision-report-v1",
            "feature_manifest_hash": self.manifest.manifest_hash,
            "fields": fields,
        }
