"""Authority for surface-specific features, labels, and value weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TaskSpec:
    name: str
    base_logit: float
    drivers: tuple[tuple[str, float], ...]
    value_weight: float


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    candidate: str
    features: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]
    primary_metric: str
    negative_definition: str

    def __post_init__(self) -> None:
        feature_set = set(self.features)
        task_names = {task.name for task in self.tasks}
        if len(task_names) != len(self.tasks):
            raise ValueError(f"duplicate task in {self.name}")
        if self.primary_metric not in task_names:
            raise ValueError(f"unknown primary metric in {self.name}")
        for task in self.tasks:
            missing = {name for name, _ in task.drivers} - feature_set
            if missing:
                raise ValueError(f"unknown drivers in {self.name}: {sorted(missing)}")

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)

    @property
    def value_weights(self) -> Mapping[str, float]:
        return {task.name: task.value_weight for task in self.tasks}


SURFACE_SPECS = {
    "feed_poi_video": SurfaceSpec(
        name="feed_poi_video",
        candidate="POI-anchored video extracted from the main Feed candidate stream",
        features=(
            "main_feed_score", "viewer_video_affinity", "poi_relevance",
            "anchor_quality", "author_quality", "viewer_poi_history",
            "sequence_affinity", "freshness", "duration_match", "risk",
        ),
        tasks=(
            TaskSpec("long_view", -0.7, (("main_feed_score", 1.1), ("viewer_video_affinity", 0.9), ("duration_match", 0.4)), 0.30),
            TaskSpec("anchor_click", -2.0, (("poi_relevance", 1.5), ("viewer_poi_history", 0.8), ("anchor_quality", 0.6)), 0.20),
            TaskSpec("detail_view", -2.7, (("anchor_quality", 0.7), ("poi_relevance", 1.2), ("sequence_affinity", 0.8)), 0.15),
            TaskSpec("favorite", -3.2, (("poi_relevance", 0.9), ("author_quality", 0.6)), 0.10),
            TaskSpec("order", -4.5, (("poi_relevance", 1.0), ("viewer_poi_history", 0.9), ("sequence_affinity", 0.8)), 0.20),
            TaskSpec("negative_feedback", -3.0, (("risk", 1.6), ("poi_relevance", -0.7)), -0.15),
        ),
        primary_metric="long_view",
        negative_definition="shown video with matured action windows and no target action",
    ),
    "poi_map_detail": SurfaceSpec(
        name="poi_map_detail",
        candidate="nearby or intent-matched POI on map and detail surfaces",
        features=("distance", "open_now", "query_match", "category_affinity", "quality", "price_match", "availability", "history"),
        tasks=(
            TaskSpec("detail_click", -1.3, (("distance", -1.0), ("query_match", 1.2), ("open_now", 0.5)), 0.25),
            TaskSpec("route", -2.8, (("distance", -0.7), ("availability", 0.8), ("history", 0.5)), 0.25),
            TaskSpec("save", -2.9, (("quality", 1.0), ("category_affinity", 0.8)), 0.20),
            TaskSpec("order", -4.0, (("availability", 1.0), ("price_match", 0.8), ("quality", 0.7)), 0.30),
        ),
        primary_metric="detail_click",
        negative_definition="eligible displayed POI skipped after a matured map/detail session",
    ),
    "ymal": SurfaceSpec(
        name="ymal",
        candidate="related POI from co-visit, semantic, category, and geographic retrieval",
        features=("co_visit", "semantic_similarity", "category_match", "distance", "quality", "novelty", "history_overlap"),
        tasks=(
            TaskSpec("related_click", -1.5, (("co_visit", 1.0), ("semantic_similarity", 1.0), ("distance", -0.5)), 0.35),
            TaskSpec("detail_dwell", -2.0, (("semantic_similarity", 0.9), ("quality", 0.8)), 0.25),
            TaskSpec("save", -3.0, (("novelty", 0.6), ("quality", 0.8), ("history_overlap", 0.5)), 0.20),
            TaskSpec("order", -4.2, (("co_visit", 0.8), ("category_match", 0.7), ("quality", 0.7)), 0.20),
        ),
        primary_metric="related_click",
        negative_definition="displayed related POI skipped within the module impression",
    ),
    "product": SurfaceSpec(
        name="product",
        candidate="available product or SKU attached to a POI, merchant, or video",
        features=("user_product_affinity", "video_product_match", "price_match", "merchant_quality", "inventory", "delivery", "discount", "history"),
        tasks=(
            TaskSpec("click", -1.5, (("user_product_affinity", 1.0), ("video_product_match", 0.8), ("discount", 0.4)), 0.15),
            TaskSpec("add_to_cart", -2.8, (("price_match", 0.8), ("inventory", 0.6), ("history", 0.5)), 0.20),
            TaskSpec("order", -4.0, (("merchant_quality", 0.7), ("delivery", 0.8), ("price_match", 0.7)), 0.30),
            TaskSpec("payment", -4.6, (("merchant_quality", 0.8), ("inventory", 0.6), ("delivery", 0.6)), 0.35),
        ),
        primary_metric="click",
        negative_definition="exposed product with entire-space click-to-payment labels matured",
    ),
    "review": SurfaceSpec(
        name="review",
        candidate="eligible review or comment attached to the current POI",
        features=("poi_relevance", "informativeness", "sentiment_match", "helpfulness_prior", "freshness", "author_trust", "toxicity", "duplication"),
        tasks=(
            TaskSpec("expand", -1.4, (("informativeness", 1.0), ("poi_relevance", 0.8), ("freshness", 0.3)), 0.25),
            TaskSpec("dwell", -1.8, (("informativeness", 1.0), ("sentiment_match", 0.4)), 0.25),
            TaskSpec("helpful", -2.8, (("helpfulness_prior", 0.8), ("author_trust", 0.6), ("poi_relevance", 0.7)), 0.35),
            TaskSpec("report", -4.0, (("toxicity", 1.5), ("duplication", 0.8)), -0.30),
        ),
        primary_metric="helpful",
        negative_definition="displayed review skipped or explicitly reported after label maturity",
    ),
}
