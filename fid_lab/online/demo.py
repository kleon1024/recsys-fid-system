"""Run and inspect one complete recommendation request."""

from __future__ import annotations

import json

from .catalog import make_catalog, make_request
from .pipeline import RecommendationPipeline


def main() -> None:
    catalog = make_catalog()
    request = make_request(catalog)
    result = RecommendationPipeline(catalog).recommend(request)
    output = {
        "request_id": result.request_id,
        "versions": result.artifact_versions,
        "stages": [
            {
                "stage": trace.stage,
                "input": trace.input_count,
                "output": trace.output_count,
                "latency_ms": trace.latency_ms,
            }
            for trace in result.traces
        ],
        "items": [
            {
                "rank": rank,
                "item_id": candidate.item_id,
                "type": catalog.get(candidate.item_id).content_type,
                "category": catalog.get(candidate.item_id).category,
                "score": round(candidate.final_score, 6),
                "recall": candidate.recall_reasons,
            }
            for rank, candidate in enumerate(result.items, start=1)
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
