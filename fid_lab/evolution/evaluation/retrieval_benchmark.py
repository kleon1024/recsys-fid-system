"""Leakage-free, equal-corpus retrieval benchmark with corrected negatives."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from ..models.retrieval import MultiInterestTwoTower, TwoTowerRetriever


NEGATIVE_MIX = {"in_batch": 0.60, "hard": 0.25, "random": 0.15}
NEGATIVES_PER_QUERY = 20


@dataclass(frozen=True)
class RetrievalWorld:
    items: np.ndarray
    category: np.ndarray
    popularity: np.ndarray
    queries: np.ndarray
    targets: np.ndarray
    history_category: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _build_world(seed: int, items: int, queries: int, dimensions: int = 24) -> RetrievalWorld:
    rng = np.random.default_rng(seed)
    categories = min(50, max(8, items // 20))
    centers = _normalize(rng.normal(size=(categories, dimensions)).astype(np.float32))
    category = rng.integers(categories, size=items)
    item_features = _normalize(
        centers[category] + rng.normal(0.0, 0.32, size=(items, dimensions))
    ).astype(np.float32)
    popularity = rng.pareto(1.4, size=items).astype(np.float32)
    popularity /= popularity.max()
    first = rng.integers(categories, size=queries)
    second = rng.integers(categories, size=queries)
    second = np.where(second == first, (second + 1) % categories, second)
    target_interest = np.where(rng.random(queries) < 0.58, first, second)
    targets = np.empty(queries, dtype=np.int64)
    for row, interest in enumerate(target_interest):
        pool = np.flatnonzero(category == interest)
        weight = 0.15 + popularity[pool]
        targets[row] = rng.choice(pool, p=weight / weight.sum())
    query_features = _normalize(
        0.62 * centers[first]
        + 0.48 * centers[second]
        + rng.normal(0.0, 0.24, size=(queries, dimensions))
    ).astype(np.float32)
    return RetrievalWorld(
        item_features,
        category,
        popularity,
        query_features,
        targets,
        first,
    )


def _split_sizes(queries: int) -> tuple[int, int]:
    train_end = max(1, int(queries * 0.70))
    validation_end = max(train_end + 1, int(queries * 0.85))
    return min(train_end, queries - 2), min(validation_end, queries - 1)


def _top_k(scores: torch.Tensor, top_k: int) -> np.ndarray:
    return torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()


def _retrieval_metrics(
    indices: np.ndarray,
    target: np.ndarray,
    popularity: np.ndarray,
) -> dict[str, float]:
    matches = indices == target[:, None]
    hit = matches.any(axis=1)
    rank = np.where(hit, matches.argmax(axis=1) + 1, 0)
    discount = np.zeros(len(rank), dtype=np.float64)
    discount[hit] = 1.0 / np.log2(rank[hit] + 1)
    long_tail = popularity[target] <= np.quantile(popularity, 0.50)
    return {
        "recall_at_k": float(hit.mean()),
        "ndcg_at_k": float(discount.mean()),
        "long_tail_recall_at_k": float(hit[long_tail].mean()) if long_tail.any() else 0.0,
        "catalog_coverage": float(len(np.unique(indices)) / len(popularity)),
        "duplicate_rate": float(
            np.mean([1.0 - len(np.unique(row)) / len(row) for row in indices])
        ),
    }


def _negative_ids(
    world: RetrievalWorld,
    target: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    batch = len(target)
    counts = (12, 5, 3)
    negative = np.empty((batch, NEGATIVES_PER_QUERY), dtype=np.int64)
    probabilities = np.empty_like(negative, dtype=np.float32)
    for row, positive in enumerate(target):
        peers = target[target != positive]
        if not len(peers):
            peers = np.flatnonzero(np.arange(len(world.items)) != positive)
        in_batch = rng.choice(peers, counts[0], replace=len(peers) < counts[0])
        hard_pool = np.flatnonzero(
            (world.category == world.category[positive])
            & (np.arange(len(world.items)) != positive)
        )
        hard = rng.choice(hard_pool, counts[1], replace=len(hard_pool) < counts[1])
        random_pool = np.flatnonzero(np.arange(len(world.items)) != positive)
        random = rng.choice(random_pool, counts[2], replace=False)
        negative[row] = np.concatenate((in_batch, hard, random))
        probabilities[row] = np.concatenate(
            (
                np.full(counts[0], NEGATIVE_MIX["in_batch"] * counts[0] / len(peers)),
                np.full(counts[1], NEGATIVE_MIX["hard"] * counts[1] / len(hard_pool)),
                np.full(counts[2], NEGATIVE_MIX["random"] * counts[2] / len(random_pool)),
            )
        )
    return negative, np.clip(probabilities, 1e-8, 1.0)


def _train_tower(
    model,
    world: RetrievalWorld,
    train_end: int,
    validation_end: int,
    device: torch.device,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 91)
    losses = []
    started = perf_counter()
    for _ in range(epochs):
        order = rng.permutation(train_end)
        epoch_loss = []
        for start in range(0, train_end, 256):
            index = order[start : start + 256]
            target = world.targets[index]
            negative, probability = _negative_ids(world, target, rng)
            query = torch.from_numpy(world.queries[index]).to(device)
            positive = torch.from_numpy(world.items[target]).to(device)
            negative_tensor = torch.from_numpy(world.items[negative]).to(device)
            if isinstance(model, MultiInterestTwoTower):
                query_state = model.encode_interests(query)
                positive_score = torch.einsum(
                    "bkd,bd->bk", query_state, model.encode_item(positive)
                ).max(dim=1).values
                negative_state = model.encode_item(
                    negative_tensor.reshape(-1, world.items.shape[1])
                ).reshape(len(index), NEGATIVES_PER_QUERY, -1)
                negative_score = torch.einsum(
                    "bkd,bnd->bkn", query_state, negative_state
                ).max(dim=1).values
            else:
                query_state = model.encode_query(query)
                positive_score = (query_state * model.encode_item(positive)).sum(dim=1)
                negative_state = model.encode_item(
                    negative_tensor.reshape(-1, world.items.shape[1])
                ).reshape(len(index), NEGATIVES_PER_QUERY, -1)
                negative_score = torch.einsum("bd,bnd->bn", query_state, negative_state)
            correction = torch.from_numpy(np.log(probability)).to(device)
            logits = torch.cat(
                (positive_score[:, None] / 0.08, negative_score / 0.08 - correction),
                dim=1,
            )
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(index), dtype=torch.long, device=device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
    with torch.no_grad():
        validation = _model_indices(
            model,
            world.queries[train_end:validation_end],
            world.items,
            min(20, len(world.items)),
            device,
        )
    validation_recall = _retrieval_metrics(
        validation,
        world.targets[train_end:validation_end],
        world.popularity,
    )["recall_at_k"]
    return {
        "seconds": perf_counter() - started,
        "loss": losses,
        "validation_recall_at_20": validation_recall,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _model_indices(model, queries, items, top_k, device) -> np.ndarray:
    with torch.no_grad():
        item_state = model.encode_item(torch.from_numpy(items).to(device))
        query = torch.from_numpy(queries).to(device)
        if isinstance(model, MultiInterestTwoTower):
            interests = model.encode_interests(query)
            scores = torch.einsum("bkd,nd->bkn", interests, item_state).max(dim=1).values
        else:
            scores = model.encode_query(query) @ item_state.T
        return _top_k(scores, top_k)


def run_retrieval_benchmark(
    seed: int = 20260823,
    items: int = 5_000,
    queries: int = 2_000,
    top_k: int = 20,
    device: str = "cpu",
    epochs: int = 5,
) -> dict[str, object]:
    if items <= top_k or queries < 10:
        raise ValueError("benchmark needs items > top_k and at least ten queries")
    if device == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    world = _build_world(seed, items, queries)
    train_end, validation_end = _split_sizes(queries)
    test_queries = world.queries[validation_end:]
    test_targets = world.targets[validation_end:]
    popular = np.argsort(-world.popularity)[:top_k]
    popular_indices = np.broadcast_to(popular, (len(test_targets), top_k))
    graph_rows = []
    for category in world.history_category[validation_end:]:
        pool = np.flatnonzero(world.category == category)
        related = pool[np.argsort(-world.popularity[pool])].tolist()
        related.extend(int(item) for item in popular if item not in related)
        graph_rows.append(related[:top_k])
    graph_indices = np.asarray(graph_rows, dtype=np.int64)
    started = perf_counter()
    content_scores = torch.from_numpy(test_queries).to(target_device) @ torch.from_numpy(
        world.items
    ).to(target_device).T
    content_indices = _top_k(content_scores, top_k)
    exact_latency = (perf_counter() - started) * 1_000.0 / len(test_targets)
    results = {
        "popular": _retrieval_metrics(popular_indices, test_targets, world.popularity),
        "co_visit_graph": _retrieval_metrics(graph_indices, test_targets, world.popularity),
        "exact_content": {
            **_retrieval_metrics(content_indices, test_targets, world.popularity),
            "milliseconds_per_query": exact_latency,
        },
    }
    models = {
        "two_tower": TwoTowerRetriever(24, 24),
        "multi_interest": MultiInterestTwoTower(24, 24),
    }
    for offset, (name, model) in enumerate(models.items()):
        training = _train_tower(
            model,
            world,
            train_end,
            validation_end,
            target_device,
            epochs,
            seed + offset,
        )
        started = perf_counter()
        indices = _model_indices(model, test_queries, world.items, top_k, target_device)
        latency = (perf_counter() - started) * 1_000.0 / len(test_targets)
        results[name] = {
            **_retrieval_metrics(indices, test_targets, world.popularity),
            "milliseconds_per_query": latency,
            "training": training,
        }
    return {
        "items": items,
        "queries": queries,
        "top_k": top_k,
        "device": str(target_device),
        "split_contract": {
            "train_queries": train_end,
            "validation_queries": validation_end - train_end,
            "test_queries": queries - validation_end,
            "query_disjoint": True,
            "frozen_item_corpus": True,
        },
        "negative_sampling": {
            "negatives_per_query": NEGATIVES_PER_QUERY,
            "source_fractions": NEGATIVE_MIX,
            "log_q_correction": True,
        },
        "models": results,
    }
