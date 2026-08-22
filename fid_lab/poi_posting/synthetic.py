"""Deterministic impression log with geographic, semantic, and sparse labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..media.materializer import MediaAsset, MediaFeatureMaterializer
from .contracts import PERMISSIONS, PoiPostingConfig, PostingBatch


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


@dataclass(frozen=True)
class Catalog:
    city: np.ndarray
    category: np.ndarray
    coordinates: np.ndarray
    popularity: np.ndarray
    semantic: np.ndarray


def make_catalog(config: PoiPostingConfig, rng: np.random.Generator) -> Catalog:
    city = rng.integers(config.cities, size=config.pois)
    category = rng.integers(config.categories, size=config.pois)
    city_centers = rng.uniform(-3.0, 3.0, size=(config.cities, 2))
    coordinates = city_centers[city] + rng.normal(0.0, 0.35, size=(config.pois, 2))
    category_basis = normalize(
        rng.normal(size=(config.categories, config.raw_semantic_dim))
    )
    semantic = normalize(
        category_basis[category]
        + rng.normal(0.0, 0.35, size=(config.pois, config.raw_semantic_dim))
    )
    popularity = np.clip(rng.pareto(2.2, size=config.pois) / 4.0, 0.0, 1.0)
    return Catalog(city, category, coordinates, popularity, semantic)


def candidate_ids(
    target: int,
    draft: np.ndarray,
    catalog: Catalog,
    config: PoiPostingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    similarity = catalog.semantic @ draft
    semantic_pool = np.argsort(-similarity)[:24]
    geo_pool = np.flatnonzero(catalog.city == catalog.city[target])
    pool = np.unique(np.concatenate(([target], semantic_pool, geo_pool)))
    selected = [target]
    choices = pool[pool != target]
    if len(choices):
        take = min(config.candidates_per_session - 1, len(choices))
        selected.extend(rng.choice(choices, size=take, replace=False).tolist())
    while len(selected) < config.candidates_per_session:
        value = int(rng.integers(config.pois))
        if value not in selected:
            selected.append(value)
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def location_context(
    target: int,
    permission: int,
    catalog: Catalog,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    target_city = int(catalog.city[target])
    if PERMISSIONS[permission] == "precise":
        return catalog.coordinates[target] + rng.normal(0.0, 0.08, size=2), target_city
    same_city = catalog.coordinates[catalog.city == target_city]
    if PERMISSIONS[permission] == "coarse":
        return same_city.mean(axis=0), target_city
    observed_city = target_city if rng.random() > 0.12 else int(rng.integers(catalog.city.max() + 1))
    return catalog.coordinates[catalog.city == observed_city].mean(axis=0), observed_city


def choose_candidate(
    candidates: np.ndarray,
    draft: np.ndarray,
    location: np.ndarray,
    author_preferences: np.ndarray,
    catalog: Catalog,
    rng: np.random.Generator,
) -> int | None:
    semantics = catalog.semantic[candidates] @ draft
    distances = np.linalg.norm(catalog.coordinates[candidates] - location, axis=1)
    preferences = author_preferences[catalog.category[candidates]]
    utilities = (
        2.4 * semantics
        - 0.7 * distances
        + 0.6 * preferences
        + 0.45 * catalog.popularity[candidates]
    )
    if rng.random() < 0.18:
        return None
    probabilities = np.exp(utilities - utilities.max())
    probabilities /= probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


def append_candidate(
    rows: dict[str, list[object]],
    session: int,
    author: int,
    poi: int,
    target: int,
    selected: int | None,
    published: bool,
    permission: int,
    observed_city: int,
    location: np.ndarray,
    frames: np.ndarray,
    text: np.ndarray,
    content: np.ndarray,
    frame_attention: np.ndarray,
    draft: np.ndarray,
    preferences: np.ndarray,
    catalog: Catalog,
) -> None:
    distance = float(np.linalg.norm(catalog.coordinates[poi] - location))
    semantic_similarity = float(catalog.semantic[poi] @ draft)
    relevant = float(
        poi == target
        or (
            catalog.category[poi] == catalog.category[target]
            and catalog.semantic[poi] @ catalog.semantic[target] > 0.78
        )
    )
    select_label = float(poi == selected)
    hard = float(
        not select_label
        and (
            catalog.city[poi] == observed_city
            or catalog.category[poi] == catalog.category[target]
            or semantic_similarity > 0.62
        )
    )
    values = {
        "session_id": session,
        "event_time": 1_700_000_000 + session * 60,
        "author_id": author,
        "poi_id": int(poi),
        "city_id": int(catalog.city[poi]),
        "category_id": int(catalog.category[poi]),
        "permission_id": permission,
        "frame_features": frames,
        "text_features": text,
        "content_features": content,
        "frame_attention": frame_attention,
        "poi_features": catalog.semantic[poi],
        "numeric_features": (
            np.log1p(distance),
            float(catalog.popularity[poi]),
            float(preferences[catalog.category[poi]]),
            float(catalog.city[poi] == observed_city),
        ),
        "labels": (select_label, float(poi == selected and published), relevant),
        "label_masks": (1.0, 1.0, 1.0),
        "hard_negative": hard,
    }
    for name, value in values.items():
        rows[name].append(value)


def build_dataset(config: PoiPostingConfig = PoiPostingConfig()) -> PostingBatch:
    rng = np.random.default_rng(config.seed)
    catalog = make_catalog(config, rng)
    author_city = rng.integers(config.cities, size=config.authors)
    author_preferences = rng.normal(0.0, 0.7, size=(config.authors, config.categories))
    materializer = MediaFeatureMaterializer(
        config.raw_semantic_dim,
        config.representation_dim,
        version="posting-media-v1",
        seed=config.seed,
    )
    rows: dict[str, list[object]] = {
        name: [] for name in PostingBatch.__dataclass_fields__
    }
    for session in range(config.sessions):
        author = int(rng.integers(config.authors))
        home_candidates = np.flatnonzero(catalog.city == author_city[author])
        target = int(
            rng.choice(home_candidates if rng.random() < 0.78 else np.arange(config.pois))
        )
        draft = normalize(
            catalog.semantic[target]
            + rng.normal(0.0, 0.28, config.raw_semantic_dim)
        )
        text = normalize(draft + rng.normal(0.0, 0.38, config.raw_semantic_dim))
        frames = normalize(
            rng.normal(
                0.0,
                1.0,
                size=(config.frames_per_draft, config.raw_semantic_dim),
            )
        )
        informative = rng.choice(config.frames_per_draft, size=2, replace=False)
        frames[informative] = normalize(
            draft
            + rng.normal(0.0, 0.32, size=(2, config.raw_semantic_dim))
        )
        media = materializer.materialize(
            MediaAsset(session, frames.astype(np.float32), text.astype(np.float32), session)
        )
        permission = int(rng.choice(3, p=(0.56, 0.29, 0.15)))
        location, observed_city = location_context(target, permission, catalog, rng)
        candidates = candidate_ids(target, draft, catalog, config, rng)
        selected = choose_candidate(
            candidates,
            draft,
            location,
            author_preferences[author],
            catalog,
            rng,
        )
        selected_relevance = 0.0 if selected is None else float(
            catalog.semantic[selected] @ catalog.semantic[target]
        )
        published = selected is not None and rng.random() < sigmoid(
            -1.8 + 2.0 * selected_relevance + 0.5 * float(selected == target)
        )
        for poi in candidates:
            append_candidate(
                rows,
                session,
                author,
                int(poi),
                target,
                selected,
                published,
                permission,
                observed_city,
                location,
                frames,
                text,
                media.content_embedding,
                media.frame_attention,
                draft,
                author_preferences[author],
                catalog,
            )
    integer = {
        "session_id",
        "event_time",
        "author_id",
        "poi_id",
        "city_id",
        "category_id",
        "permission_id",
    }
    arrays = {
        name: np.asarray(values, dtype=np.int64 if name in integer else np.float32)
        for name, values in rows.items()
    }
    return PostingBatch(**arrays)
