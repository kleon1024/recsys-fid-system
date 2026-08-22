"""Equal-budget heuristic, FAISS, two-tower, and multi-interest recall comparison."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

from ..models.retrieval import MultiInterestTwoTower, TwoTowerRetriever


def _recall_at_k(indices: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean([value in row for value, row in zip(target, indices)]))


def _graph_candidates(
    category: np.ndarray,
    target_category: np.ndarray,
    popular: np.ndarray,
    top_k: int,
) -> np.ndarray:
    rows = []
    for value in target_category:
        related = np.flatnonzero(category == value).tolist()
        related.extend(item for item in popular.tolist() if item not in related)
        rows.append(related[:top_k])
    return np.asarray(rows, dtype=np.int64)


def _train_tower(model, queries, items, targets, device: str, epochs: int) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    rng = np.random.default_rng(91)
    for _ in range(epochs):
        for start in range(0, len(queries), 256):
            index = rng.integers(0, len(queries), size=min(256, len(queries)))
            query = torch.from_numpy(queries[index]).to(device)
            item = torch.from_numpy(items[targets[index]]).to(device)
            scores = model(query, item)
            loss = torch.nn.functional.cross_entropy(
                scores, torch.arange(len(index), device=device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_retrieval_benchmark(
    seed: int = 20260823,
    items: int = 5_000,
    queries: int = 2_000,
    top_k: int = 20,
    device: str = "cpu",
) -> dict[str, object]:
    if device == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    item_features = rng.normal(size=(items, 24)).astype(np.float32)
    item_features /= np.linalg.norm(item_features, axis=1, keepdims=True)
    target = rng.integers(items, size=queries)
    query_features = item_features[target] + rng.normal(0.0, 0.18, size=(queries, 24))
    query_features = query_features.astype(np.float32)
    category = rng.integers(50, size=items)
    popularity = rng.pareto(1.4, size=items)
    popular = np.argsort(-popularity)[:top_k]
    popular_indices = np.broadcast_to(popular, (queries, top_k))
    target_category = category[target]
    graph_indices = _graph_candidates(category, target_category, popular, top_k)
    started = perf_counter()
    index = NearestNeighbors(n_neighbors=top_k, algorithm="brute", metric="cosine")
    index.fit(item_features)
    _, content_indices = index.kneighbors(query_features)
    content_ms = (perf_counter() - started) * 1_000.0 / queries
    models = {
        "two_tower": TwoTowerRetriever(24, 24),
        "multi_interest": MultiInterestTwoTower(24, 24),
    }
    results = {
        "popular": {"recall_at_k": _recall_at_k(popular_indices, target)},
        "co_visit_graph": {"recall_at_k": _recall_at_k(graph_indices, target)},
        "exact_content": {
            "recall_at_k": _recall_at_k(content_indices, target),
            "milliseconds_per_query": content_ms,
        },
    }
    for name, model in models.items():
        _train_tower(model, query_features, item_features, target, device, epochs=3)
        with torch.no_grad():
            item_state = model.encode_item(torch.from_numpy(item_features).to(device)).cpu().numpy()
            started = perf_counter()
            if name == "multi_interest":
                interests = model.encode_interests(
                    torch.from_numpy(query_features).to(device)
                ).cpu().numpy()
                scores = np.einsum("bkd,nd->bkn", interests, item_state).max(axis=1)
                indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
            else:
                query_state = model.encode_query(torch.from_numpy(query_features).to(device))
                query_state = query_state.cpu().numpy()
                scores = query_state @ item_state.T
                indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        results[name] = {
            "recall_at_k": _recall_at_k(indices, target),
            "milliseconds_per_query": (perf_counter() - started) * 1_000.0 / queries,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    return {"items": items, "queries": queries, "top_k": top_k, "models": results}
