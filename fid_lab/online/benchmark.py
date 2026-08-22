"""Multi-request acceptance smoke for latency, fill, safety, and diversity."""

from __future__ import annotations

from collections import Counter
import json

import numpy as np

from .catalog import make_catalog, make_request
from .pipeline import RecommendationPipeline


def main() -> None:
    catalog = make_catalog()
    pipeline = RecommendationPipeline(catalog)
    results = [pipeline.recommend(make_request(catalog, user_id=user_id)) for user_id in range(100)]
    stage_latencies: dict[str, list[float]] = {}
    category_counts: list[int] = []
    unsafe = 0
    duplicate_slates = 0
    type_counts: Counter[str] = Counter()
    for result in results:
        ids = [candidate.item_id for candidate in result.items]
        duplicate_slates += int(len(ids) != len(set(ids)))
        categories = {catalog.get(item_id).category for item_id in ids}
        category_counts.append(len(categories))
        for item_id in ids:
            item = catalog.get(item_id)
            unsafe += int(not item.is_safe or not item.is_active)
            type_counts[item.content_type] += 1
        for trace in result.traces:
            stage_latencies.setdefault(trace.stage, []).append(trace.latency_ms)
    output = {
        "requests": len(results),
        "full_slate_rate": sum(len(result.items) == 20 for result in results) / len(results),
        "unsafe_items": unsafe,
        "slates_with_duplicates": duplicate_slates,
        "mean_categories_per_slate": round(float(np.mean(category_counts)), 3),
        "content_types": dict(type_counts),
        "latency_ms": {
            stage: {
                "p50": round(float(np.percentile(values, 50)), 3),
                "p95": round(float(np.percentile(values, 95)), 3),
            }
            for stage, values in stage_latencies.items()
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
