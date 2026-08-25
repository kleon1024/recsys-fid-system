"""One declarative contract for every observable serving feature and FID."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


FEATURE_GROUPS = frozenset({
    "user", "item", "creator", "context", "route", "counter",
    "sequence", "content", "cross",
})


@dataclass(frozen=True)
class FeatureField:
    name: str
    tensor: str
    group: str
    source: str
    transform: str
    dtype: str
    default: float | int
    ttl_ticks: int
    namespace: str = ""
    slot: int = 0
    buckets: int = 0
    vocabulary: int = 0

    def __post_init__(self) -> None:
        if self.tensor not in {"dense", "sparse"}:
            raise ValueError(f"unsupported feature tensor: {self.tensor}")
        if self.group not in FEATURE_GROUPS:
            raise ValueError(f"unsupported feature group: {self.group}")
        if self.ttl_ticks < -1:
            raise ValueError("feature TTL must be -1 or nonnegative")
        if self.tensor == "sparse" and (
            not self.namespace or self.slot <= 0 or self.buckets <= 1
        ):
            raise ValueError("sparse features require namespace, slot and buckets")
        if self.tensor == "dense" and any((self.slot, self.buckets)):
            raise ValueError("dense features cannot own FID slots or buckets")


@dataclass(frozen=True)
class FeatureManifest:
    schema_version: str
    fid_version: str
    fields: tuple[FeatureField, ...]

    def __post_init__(self) -> None:
        if not self.schema_version or not self.fid_version or not self.fields:
            raise ValueError("feature manifest identity is incomplete")
        names = tuple(field.name for field in self.fields)
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        slots = tuple(
            field.slot for field in self.fields if field.tensor == "sparse"
        )
        if len(slots) != len(set(slots)):
            raise ValueError("sparse feature slots must be unique")

    @property
    def dense_fields(self) -> tuple[FeatureField, ...]:
        return tuple(field for field in self.fields if field.tensor == "dense")

    @property
    def sparse_fields(self) -> tuple[FeatureField, ...]:
        return tuple(field for field in self.fields if field.tensor == "sparse")

    @property
    def dense_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.dense_fields)

    @property
    def sparse_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.sparse_fields)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fid_version": self.fid_version,
            "fields": [asdict(field) for field in self.fields],
        }

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _dense(
    name: str,
    group: str,
    source: str,
    transform: str,
    *,
    ttl_ticks: int = -1,
) -> FeatureField:
    return FeatureField(
        name, "dense", group, source, transform, "float32", 0.0, ttl_ticks,
    )


def _sparse(
    name: str,
    group: str,
    source: str,
    transform: str,
    slot: int,
    buckets: int,
    vocabulary: int = 0,
) -> FeatureField:
    return FeatureField(
        name, "sparse", group, source, transform, "int64", 0, -1,
        f"v4:{name}", slot, buckets, vocabulary,
    )


DEFAULT_FEATURE_MANIFEST = FeatureManifest(
    schema_version="observable-feature-manifest-v1",
    fid_version="v2",
    fields=(
        _dense("semantic_affinity", "content", "query_embedding,item_embedding", "dot"),
        _dense("quality_prior", "content", "catalog.quality_prior", "identity"),
        _dense("item_impression_log", "counter", "item.impression_count", "log1p/8", ttl_ticks=1),
        _dense("item_engagement_rate", "counter", "item.long_view,item.click,item.impression", "clip((long_view+click)/max(impression,1),0,1)", ttl_ticks=1),
        _dense(
            "freshness", "item", "request.event_time,item.publish_time",
            "exp(-age_ticks/(2*ticks_per_day))",
        ),
        _dense("geo_match", "context", "user.country,user.region,item.country,item.region", "0.35*country+0.65*region"),
        _dense("inventory", "item", "projection.item_inventory", "identity", ttl_ticks=1),
        _dense("recall_score_scaled", "route", "retrieval.recall_score", "max(score,0)*20"),
        _dense("creator_engagement_rate", "creator", "creator.engagements,creator.impressions", "clip(engagements/max(impressions,1),0,1)", ttl_ticks=1),
        _dense("history_repeat", "sequence", "user.history_item_id,item_id", "any_equal"),
        _dense("sequence_affinity", "sequence", "user.history_embedding,item_embedding", "candidate_attention_dot"),
        _sparse("user_id", "user", "request.user_id", "identity", 1, 1 << 18),
        _sparse("item_id", "item", "candidate.item_id", "identity", 101, 1 << 20),
        _sparse("creator_id", "creator", "candidate.creator_id", "identity", 102, 1 << 18),
        _sparse("content_kind", "content", "catalog.content_kind", "identity", 103, 32, 16),
        _sparse("topic_id", "content", "catalog.topic_id", "identity", 104, 1 << 15),
        _sparse("surface", "context", "request.surface", "identity", 201, 32, 16),
        _sparse("user_country", "context", "projection.user_country", "identity", 202, 1 << 10),
        _sparse("user_region", "context", "projection.user_region", "identity", 203, 1 << 14),
        _sparse("request_utc_hour", "context", "request.event_time", "tick_mod_day_to_hour", 204, 32, 24),
        _sparse("route_bits", "route", "retrieval.route_bits", "identity", 205, 1 << 12),
        _sparse("lifecycle_id", "item", "projection.item_lifecycle", "identity", 206, 32, 16),
        _sparse("user_x_kind", "cross", "request.user_id,catalog.content_kind", "user_id*16+content_kind", 301, 1 << 20),
        _sparse("surface_x_kind", "cross", "request.surface,catalog.content_kind", "surface*16+content_kind", 302, 512),
    ),
)
