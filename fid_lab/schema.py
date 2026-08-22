"""Single authority for feature slots, ownership groups, and explicit crosses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .fid import FidCodec


VALID_GROUPS = frozenset({"user", "item", "context", "cross"})


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    slot: int
    group: str
    sources: tuple[str, ...]
    buckets: int = 512

    def __post_init__(self) -> None:
        if self.group not in VALID_GROUPS:
            raise ValueError(f"unknown feature group: {self.group}")
        if self.buckets <= 1:
            raise ValueError("buckets must exceed one")
        if self.group == "cross" and len(self.sources) < 2:
            raise ValueError("cross features require at least two source fields")
        if self.group != "cross" and self.sources != (self.name,):
            raise ValueError("non-cross features must name themselves as their source")


class FeatureRegistry:
    def __init__(self, specs: Sequence[FeatureSpec]) -> None:
        self.specs = tuple(specs)
        names = [spec.name for spec in self.specs]
        slots = [spec.slot for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        if len(slots) != len(set(slots)):
            raise ValueError("feature slots must be unique")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def indices_by_group(self, group: str) -> tuple[int, ...]:
        return tuple(i for i, spec in enumerate(self.specs) if spec.group == group)

    def raw_value(self, spec: FeatureSpec, row: Mapping[str, object]) -> object:
        if spec.group != "cross":
            return row[spec.name]
        # Length-prefix each component so ('ab', 'c') cannot collide with ('a', 'bc').
        return "|".join(f"{len(str(row[name]))}:{row[name]}" for name in spec.sources)

    def encode_row(
        self, row: Mapping[str, object], codec: FidCodec
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        fids: list[int] = []
        bucket_ids: list[int] = []
        for spec in self.specs:
            raw_value = self.raw_value(spec, row)
            fid = codec.encode(spec.slot, spec.name, raw_value)
            _, signature = codec.unpack(fid)
            fids.append(fid)
            bucket_ids.append(signature % spec.buckets)
        return tuple(fids), tuple(bucket_ids)


DEFAULT_SCHEMA = FeatureRegistry(
    [
        FeatureSpec("user_id", 1, "user", ("user_id",), 256),
        FeatureSpec("age_bucket", 2, "user", ("age_bucket",), 16),
        FeatureSpec("item_id", 101, "item", ("item_id",), 384),
        FeatureSpec("category", 102, "item", ("category",), 32),
        FeatureSpec("country", 201, "context", ("country",), 16),
        FeatureSpec("device", 202, "context", ("device",), 16),
        FeatureSpec("hour_bucket", 203, "context", ("hour_bucket",), 8),
        FeatureSpec("category_x_device", 301, "cross", ("category", "device"), 128),
        FeatureSpec("country_x_category", 302, "cross", ("country", "category"), 128),
    ]
)
