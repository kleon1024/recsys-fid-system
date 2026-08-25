"""Typed persisted inputs for v4 retrieval training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class RetrievalFeatureContract:
    version: str = "v4-retrieval-feature-contract-v1"
    counter_log_scale: float = 8.0
    daily_ticks: int = 96
    query_sparse_fields: tuple[str, ...] = (
        "user_id_hash", "surface", "query_topic", "user_country", "user_region",
    )
    query_dense_fields: tuple[str, ...] = (
        "log1p_user_event_counts", "log1p_user_surface_counts",
        "request_time_sin", "request_time_cos", "history_content_state",
    )
    item_sparse_fields: tuple[str, ...] = (
        "item_id_hash", "content_kind", "topic_id", "creator_id_hash",
        "item_country", "item_region",
    )
    item_dense_fields: tuple[str, ...] = (
        "content_embedding", "quality_prior", "log_duration", "publish_time",
    )
    history_fields: tuple[str, ...] = (
        "history_item_id", "history_event_type", "point_in_time_order",
    )

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_RETRIEVAL_FEATURE_CONTRACT = RetrievalFeatureContract()


def tensor_content_hash(tensors: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RetrievalCorpus:
    item_id: torch.Tensor
    content_kind: torch.Tensor
    topic_id: torch.Tensor
    content_embedding: torch.Tensor
    creator_id: torch.Tensor
    country: torch.Tensor
    region: torch.Tensor
    publish_time: torch.Tensor
    quality_prior: torch.Tensor
    duration_seconds: torch.Tensor
    active: torch.Tensor
    catalog_version: str

    def __post_init__(self) -> None:
        items = len(self.item_id)
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "catalog_version":
                continue
            if field.name == "content_embedding":
                if value.ndim != 2 or value.shape[0] != items:
                    raise ValueError("corpus content embeddings are not item aligned")
            elif value.shape != (items,):
                raise ValueError(f"corpus field {field.name} is not item aligned")
        if not self.catalog_version:
            raise ValueError("retrieval corpus requires a catalog version")
        if not torch.equal(self.item_id, torch.arange(items)):
            raise ValueError("retrieval corpus item IDs must be contiguous")

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "catalog_version"
        }

    @property
    def content_sha256(self) -> str:
        return tensor_content_hash(self.tensors)

    def checkpoint(self) -> dict[str, object]:
        return {
            "schema": "v4-retrieval-corpus-v1",
            "catalog_version": self.catalog_version,
            "content_sha256": self.content_sha256,
            "tensors": {
                name: value.detach().cpu().clone()
                for name, value in self.tensors.items()
            },
        }

    def save(self, path: Path) -> str:
        torch.save(self.checkpoint(), path)
        return sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> RetrievalCorpus:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("schema") != "v4-retrieval-corpus-v1":
            raise ValueError("retrieval corpus schema is unsupported")
        result = cls(catalog_version=str(value["catalog_version"]), **value["tensors"])
        if result.content_sha256 != value["content_sha256"]:
            raise ValueError("retrieval corpus content hash differs")
        return result


@dataclass(frozen=True)
class RetrievalQueryBatch:
    request_id: torch.Tensor
    user_id: torch.Tensor
    surface: torch.Tensor
    event_time: torch.Tensor
    query_topic: torch.Tensor
    user_country: torch.Tensor
    user_region: torch.Tensor
    user_event_counts: torch.Tensor
    user_surface_counts: torch.Tensor
    history_item_id: torch.Tensor
    history_event_type: torch.Tensor
    positive_item_id: torch.Tensor
    positive_strength: torch.Tensor
    negative_item_id: torch.Tensor
    negative_expected_count: torch.Tensor
    negative_loss_mask: torch.Tensor
    feature_manifest_hash: str
    partition_content_hashes: tuple[str, ...]
    event_watermark: int

    def __post_init__(self) -> None:
        rows = len(self.request_id)
        for name in (
            "user_id", "surface", "event_time", "query_topic", "user_country",
            "user_region", "positive_item_id", "positive_strength",
        ):
            if getattr(self, name).shape != (rows,):
                raise ValueError(f"retrieval {name} is not request aligned")
        for name in (
            "user_event_counts", "user_surface_counts", "history_item_id",
            "history_event_type", "negative_item_id", "negative_expected_count",
            "negative_loss_mask",
        ):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != rows:
                raise ValueError(f"retrieval {name} is not request aligned")
        if self.history_item_id.shape != self.history_event_type.shape:
            raise ValueError("retrieval history item and event tensors differ")
        if not (
            self.negative_item_id.shape == self.negative_expected_count.shape
            == self.negative_loss_mask.shape
        ):
            raise ValueError("retrieval negative tensors differ")
        if self.negative_loss_mask.any() and (
            self.negative_expected_count[self.negative_loss_mask] <= 0
        ).any():
            raise ValueError("valid retrieval negatives require expected counts")
        if not self.feature_manifest_hash or not self.partition_content_hashes:
            raise ValueError("retrieval batch lineage is incomplete")

    def select(self, selected: torch.Tensor) -> RetrievalQueryBatch:
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value[selected] if isinstance(value, torch.Tensor) else value
        return RetrievalQueryBatch(**values)


@dataclass(frozen=True)
class RetrievalModelConfig:
    architecture: str
    representation_dim: int = 32
    hidden_dim: int = 96
    embedding_dim: int = 12
    user_hash_buckets: int = 262_144
    item_hash_buckets: int = 262_144
    creator_hash_buckets: int = 131_072
    interests: int = 3
    temperature: float = 0.08
    epochs: int = 4
    batch_size: int = 1_024
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    seed: int = 2_026_082_5

    def __post_init__(self) -> None:
        if self.architecture not in {"two_tower", "multi_interest"}:
            raise ValueError("unsupported retrieval architecture")
        numeric = tuple(value for value in asdict(self).values() if not isinstance(value, str))
        if any(value <= 0 for value in numeric):
            raise ValueError("retrieval model configuration must be positive")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()
