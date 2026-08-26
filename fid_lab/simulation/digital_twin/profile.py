"""One resolved runtime contract for a continuous recommendation world."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json


@dataclass(frozen=True)
class SimulationProfile:
    name: str
    users: int
    items: int
    ticks_per_day: int
    topics: int
    countries: int
    regions_per_country: int
    embedding_dim: int
    history_length: int
    feed_exposure_history_length: int
    route_k: int
    merged_k: int
    coarse_k: int
    fine_k: int
    expose_k: int
    seed: int

    def __post_init__(self) -> None:
        dimensions = tuple(
            value for name, value in asdict(self).items() if name != "name"
        )
        if not self.name or min(dimensions) <= 0:
            raise ValueError("simulation profile fields must be positive")
        if not self.merged_k >= self.coarse_k >= self.fine_k >= self.expose_k:
            raise ValueError("simulation profile cascade budgets are inconsistent")

    def scaled(self, *, users: int, items: int) -> SimulationProfile:
        return replace(self, users=users, items=items)

    def manifest(self) -> dict[str, object]:
        return {"schema": "simulation-profile/v1", **asdict(self)}

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


STANDARD_FEED_PROFILE = SimulationProfile(
    name="feed-semantic-v2",
    users=100_000,
    items=1_000_000,
    ticks_per_day=96,
    topics=512,
    countries=12,
    regions_per_country=16,
    embedding_dim=64,
    history_length=512,
    feed_exposure_history_length=256,
    route_k=24,
    merged_k=96,
    coarse_k=48,
    fine_k=16,
    expose_k=8,
    seed=809,
)
