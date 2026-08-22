"""Compare constrained Semantic-ID generation with exact vector retrieval."""

from __future__ import annotations

import json

import numpy as np

from ..online.catalog import make_catalog, make_request
from ..online.stages.retrieval import LocalVikingIndex
from .retriever import GenerativeRetriever
from .semantic_ids import SemanticIdIndex


def main() -> None:
    catalog = make_catalog(size=600)
    item_ids = np.asarray([item.item_id for item in catalog.items])
    embeddings = np.stack([item.embedding for item in catalog.items])
    index = SemanticIdIndex.fit(item_ids, embeddings)
    request = make_request(catalog)
    generated = GenerativeRetriever(index).retrieve(request.user_embedding, limit=20)
    exact = LocalVikingIndex(catalog).recall(request, 20)
    generated_ids = {item.item_id for item in generated}
    exact_ids = {item.item_id for item in exact}
    print(
        json.dumps(
            {
                "semantic_id_version": index.version,
                "decoder_scorer": "teaching_oracle_prefix_similarity",
                "items": len(index.codes),
                "unique_codes": len(set(index.codes.values())),
                "generated": len(generated),
                "valid_generated": all(item.semantic_id in index.codes.values() for item in generated),
                "exact_top20_overlap": len(generated_ids & exact_ids) / len(exact_ids),
                "sample": [
                    {"item_id": item.item_id, "semantic_id": item.semantic_id, "score": item.score}
                    for item in generated[:3]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
