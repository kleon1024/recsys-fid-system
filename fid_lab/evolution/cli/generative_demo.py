"""Train and evaluate a constrained Semantic-ID generative recall route."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ...generative.semantic_ids import SemanticIdIndex
from ..models.generative import (
    LearnedGenerativeRetriever,
    SessionGenerator,
    train_semantic_decoder,
)


def run(device: str = "cpu", items: int = 256, epochs: int = 25) -> dict[str, object]:
    rng = np.random.default_rng(20260823)
    item_ids = np.arange(items)
    embeddings = rng.normal(size=(items, 24)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    queries = embeddings + rng.normal(0.0, 0.08, size=embeddings.shape).astype(np.float32)
    index = SemanticIdIndex.fit(item_ids, embeddings, levels=3, codebook_size=8)
    model, loss = train_semantic_decoder(index, queries, item_ids, epochs, device)
    retriever = LearnedGenerativeRetriever(index, model)
    evaluated = min(items, 12)
    hits = 0
    valid = 0
    for row in range(evaluated):
        generated = retriever.retrieve(
            torch.from_numpy(queries[row]).to(device), limit=20, beam_size=40
        )
        hits += int(int(item_ids[row]) in {value.item_id for value in generated})
        valid += int(all(value.semantic_id in index.codes.values() for value in generated))
    session = SessionGenerator(
        retriever,
        {int(item): int(item % 32) for item in item_ids},
        {int(item): int(item % 12) for item in item_ids},
    ).generate(torch.from_numpy(queries[0]).to(device), size=10)
    return {
        "implementation": "PyTorch TransformerDecoder plus existing SemanticIdIndex",
        "items": items,
        "epochs": epochs,
        "initial_loss": loss[0],
        "final_loss": loss[-1],
        "evaluated_queries": evaluated,
        "recall_at_20": hits / evaluated,
        "valid_item_rate": valid / evaluated,
        "session_size": len(session),
        "session_unique": len({value.item_id for value in session}) == len(session),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--items", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.items, args.epochs), indent=2))


if __name__ == "__main__":
    main()
