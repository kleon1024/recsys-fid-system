"""Versioned contracts for one shared multi-surface simulation world."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum


TWIN_VERSION = "multi-surface-digital-twin-v3-causal-feature-lineage"


class Surface(IntEnum):
    FEED = 0
    SEARCH = 1
    COMMERCE = 2
    LIVE = 3
    LOCAL = 4
    POSTING = 5


class ItemKind(IntEnum):
    SHORT_VIDEO = 0
    PHOTO = 1
    ARTICLE = 2
    CARD = 3
    LIVE_ROOM = 4
    PRODUCT = 5
    POI = 6
    AD = 7
    CREATOR_PROMPT = 8


ITEM_KINDS = tuple(kind.name.lower() for kind in ItemKind)


@dataclass(frozen=True)
class SurfaceContract:
    surface: Surface
    allowed_kinds: tuple[ItemKind, ...]
    response_tasks: tuple[str, ...]
    entry_event: str
    primary_metric: str
    slate_size: int


SURFACE_CONTRACTS = {
    Surface.FEED: SurfaceContract(
        Surface.FEED,
        (
            ItemKind.SHORT_VIDEO, ItemKind.PHOTO, ItemKind.ARTICLE,
            ItemKind.CARD, ItemKind.LIVE_ROOM, ItemKind.PRODUCT,
            ItemKind.POI, ItemKind.AD,
        ),
        (
            "play", "play_3s", "long_view", "complete", "like",
            "comment", "share", "follow", "click", "negative",
        ),
        "app_open_or_slide",
        "stay_seconds",
        1,
    ),
    Surface.SEARCH: SurfaceContract(
        Surface.SEARCH,
        (
            ItemKind.SHORT_VIDEO, ItemKind.ARTICLE, ItemKind.PRODUCT,
            ItemKind.POI, ItemKind.LIVE_ROOM, ItemKind.AD,
        ),
        ("click", "long_view", "detail", "order", "negative"),
        "query_submit",
        "successful_search",
        5,
    ),
    Surface.COMMERCE: SurfaceContract(
        Surface.COMMERCE,
        (ItemKind.PRODUCT, ItemKind.CARD, ItemKind.AD),
        ("click", "detail", "add_cart", "order", "payment", "negative"),
        "shelf_or_product_entry",
        "payment",
        4,
    ),
    Surface.LIVE: SurfaceContract(
        Surface.LIVE,
        (ItemKind.LIVE_ROOM, ItemKind.PRODUCT, ItemKind.AD),
        ("click", "long_view", "follow", "order", "payment", "negative"),
        "live_entry",
        "live_stay_seconds",
        1,
    ),
    Surface.LOCAL: SurfaceContract(
        Surface.LOCAL,
        (
            ItemKind.POI, ItemKind.SHORT_VIDEO, ItemKind.PHOTO,
            ItemKind.ARTICLE, ItemKind.PRODUCT,
        ),
        ("click", "detail", "favorite", "order", "payment", "negative"),
        "poi_anchor_map_or_nearby_entry",
        "poi_detail",
        4,
    ),
    Surface.POSTING: SurfaceContract(
        Surface.POSTING,
        (ItemKind.CREATOR_PROMPT, ItemKind.POI, ItemKind.PRODUCT),
        ("click", "create", "publish", "negative"),
        "camera_or_publish_entry",
        "publish",
        5,
    ),
}


@dataclass(frozen=True)
class TwinConfig:
    users: int = 1_000_000
    catalog_items: int = 2_000_000
    creators: int = 250_000
    topics: int = 24
    countries: int = 12
    regions_per_country: int = 8
    initial_registered_fraction: float = 0.82
    preperiod_steps: int = 8
    measurement_steps: int = 32
    steps_per_day: int = 8
    history_length: int = 64
    route_candidates: int = 16
    routes: int = 6
    coarse_keep: int = 48
    fine_keep: int = 12
    audit_users: int = 2_560
    training_trace_users: int = 4_096
    batch_users: int = 250_000
    serve_chunk_users: int = 50_000
    seed: int = 20260824
    environment_seed: int = 20260825
    device: str = "cuda:0"
    version: str = TWIN_VERSION

    def __post_init__(self):
        if self.version != TWIN_VERSION:
            raise ValueError("unsupported digital-twin version")
        if min(self.users, self.catalog_items, self.creators) <= 0:
            raise ValueError("population and supply sizes must be positive")
        if self.creators > self.catalog_items:
            raise ValueError("creator count cannot exceed catalog size")
        if self.regions_per_country < 1:
            raise ValueError("every country requires at least one region")
        if not 0.0 < self.initial_registered_fraction < 1.0:
            raise ValueError("initial registered fraction must be between zero and one")
        if self.history_length < 8 or self.history_length > 128:
            raise ValueError("exposure history must be between 8 and 128")
        merged = self.route_candidates * self.routes
        if not self.fine_keep <= self.coarse_keep <= merged:
            raise ValueError("candidate budgets must form recall >= coarse >= fine")
        if self.preperiod_steps < 1 or self.measurement_steps < 1:
            raise ValueError("pre-period and measurement windows are required")
        if self.steps_per_day < 1:
            raise ValueError("steps per simulated day must be positive")
        if self.training_trace_users < self.audit_users:
            raise ValueError("training trace must include the audit trace")
        if self.serve_chunk_users <= 0:
            raise ValueError("serving chunk must be positive")

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["capacity"] = {
            "surfaces": len(Surface),
            "item_kinds": len(ItemKind),
            "response_tasks": 17,
            "regions": self.countries * self.regions_per_country,
            "candidate_rows_per_request": self.routes * self.route_candidates,
            "trajectory_steps": self.preperiod_steps + self.measurement_steps,
            "stateful_exposure_slots": self.users * self.history_length,
            "training_requests_per_step": min(
                self.users, self.batch_users, self.training_trace_users
            ),
            "persistent_user_shard": self.batch_users,
            "candidate_compute_microbatch": min(
                self.batch_users, self.serve_chunk_users
            ),
            "candidate_work_units": (
                self.users
                * (self.preperiod_steps + 2 * self.measurement_steps)
                * self.routes
                * self.route_candidates
            ),
        }
        value["randomness_ownership"] = {
            "platform_seed": self.seed,
            "hidden_environment_seed": self.environment_seed,
        }
        return value


@dataclass(frozen=True)
class TwinPolicy:
    name: str
    affinity_weight: float = 1.0
    quality_weight: float = 0.35
    freshness_weight: float = 0.10
    popularity_weight: float = 0.05
    trend_weight: float = 0.12
    query_weight: float = 0.24
    merchant_weight: float = 0.08
    availability_weight: float = 0.08
    recall_weight: float = 0.10
    route_weights: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.65, 0.75)
    enabled_routes: tuple[bool, ...] = (
        False, False, True, True, False, True,
    )
    realtime_weight: float = 0.20
    geo_weight: float = 0.20
    commerce_weight: float = 0.10
    risk_penalty: float = 0.20
    recent_item_hard_filter: bool = True
    author_fatigue_penalty: float = 0.04
    cluster_fatigue_penalty: float = 0.06
    topic_fatigue_penalty: float = 0.03
    exploration_rate: float = 0.0
    ad_value_weight: float = 0.04
    local_value_weight: float = 0.04
    live_value_weight: float = 0.03
    product_value_weight: float = 0.04
    max_ads_per_history: int = 2
    min_ad_gap: int = 4
    coarse_keep: int = 48
    fine_keep: int = 12

    def __post_init__(self):
        if not self.name:
            raise ValueError("policy name is required")
        if min(self.coarse_keep, self.fine_keep) <= 0:
            raise ValueError("policy candidate budgets must be positive")
        if self.fine_keep > self.coarse_keep:
            raise ValueError("fine budget cannot exceed coarse budget")
        if len(self.route_weights) != 6 or min(self.route_weights) < 0:
            raise ValueError("six nonnegative route weights are required")
        if len(self.enabled_routes) != 6 or not any(self.enabled_routes):
            raise ValueError("at least one of six retrieval routes is required")
        if min(
            self.risk_penalty,
            self.author_fatigue_penalty,
            self.cluster_fatigue_penalty,
            self.topic_fatigue_penalty,
        ) < 0:
            raise ValueError("policy penalties must be nonnegative")
        if not 0.0 <= self.exploration_rate <= 0.25:
            raise ValueError("exploration rate must be in [0, 0.25]")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


BASELINE_POLICY = TwinPolicy(name="shared_rules_v1")
