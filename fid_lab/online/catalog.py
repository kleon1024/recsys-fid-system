"""Deterministic local catalog used by every online stage."""

from __future__ import annotations

import numpy as np

from .domain import Item, RequestContext


CATEGORIES = ("finance", "sports", "technology", "travel", "food", "culture")
CONTENT_TYPES = ("organic", "organic", "organic", "organic", "live", "ad")


class ItemCatalog:
    def __init__(self, items: list[Item], version: str = "catalog-v1") -> None:
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("item IDs must be unique")
        self.items = tuple(items)
        self.by_id = {item.item_id: item for item in items}
        self.version = version

    def get(self, item_id: int) -> Item:
        return self.by_id[item_id]


def make_catalog(size: int = 1200, dimension: int = 24, seed: int = 19) -> ItemCatalog:
    rng = np.random.default_rng(seed)
    category_centers = rng.normal(size=(len(CATEGORIES), dimension))
    category_centers /= np.linalg.norm(category_centers, axis=1, keepdims=True)
    items: list[Item] = []
    for item_id in range(size):
        category_index = item_id % len(CATEGORIES)
        embedding = category_centers[category_index] + rng.normal(0, 0.32, dimension)
        embedding = embedding.astype(np.float32)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        countries = frozenset({"SG", "US", "GB"})
        if item_id % 17 == 0:
            countries = frozenset({"SG"})
        items.append(
            Item(
                item_id=item_id,
                content_type=CONTENT_TYPES[item_id % len(CONTENT_TYPES)],
                category=CATEGORIES[category_index],
                creator_id=item_id % 160,
                embedding=embedding,
                popularity=float(rng.beta(2.2, 5.0)),
                quality=float(rng.beta(5.0, 2.0)),
                age_hours=float(rng.exponential(120.0)),
                allowed_countries=countries,
                is_safe=item_id % 113 != 0,
                is_active=item_id % 127 != 0,
            )
        )
    return ItemCatalog(items)


def make_request(
    catalog: ItemCatalog, user_id: int = 42, country: str = "SG", size: int = 20
) -> RequestContext:
    primary = CATEGORIES[user_id % len(CATEGORIES)]
    secondary = CATEGORIES[(user_id + 2) % len(CATEGORIES)]
    matching = [item.embedding for item in catalog.items if item.category == primary][:20]
    user_embedding = np.mean(matching, axis=0).astype(np.float32)
    user_embedding /= max(float(np.linalg.norm(user_embedding)), 1e-8)
    seen = frozenset(item.item_id for item in catalog.items if item.item_id % 97 == user_id % 97)
    return RequestContext(
        request_id=f"request-{user_id}",
        user_id=user_id,
        country=country,
        user_embedding=user_embedding,
        category_affinity={primary: 1.0, secondary: 0.45},
        device=user_id % 3,
        hour_bucket=(user_id // 3) % 4,
        seen_item_ids=seen,
        size=size,
    )
