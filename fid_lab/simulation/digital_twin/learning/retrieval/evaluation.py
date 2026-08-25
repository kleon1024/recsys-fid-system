"""Equal-budget retrieval metrics and fixed-ranker downstream diagnosis."""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from time import perf_counter

import numpy as np
import torch

from ...contracts import Surface
from ...platform.routes import surface_eligibility
from .artifact import RetrievalANNIndex, RetrievalArtifact
from .contracts import RetrievalCorpus, RetrievalQueryBatch


def _last_history(batch: RetrievalQueryBatch) -> np.ndarray:
    history = batch.history_item_id.numpy()
    valid = history >= 0
    location = valid.sum(axis=1) - 1
    result = np.full(len(history), -1, dtype=np.int64)
    rows = np.flatnonzero(location >= 0)
    result[rows] = history[rows, location[rows]]
    return result


def _fill_unique(primary, fallback, top_k: int) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for raw_item in chain(primary, fallback):
        if len(output) == top_k:
            break
        item = int(raw_item)
        if item < 0 or item in seen:
            continue
        output.append(item)
        seen.add(item)
    return output


def _round_robin_row(candidate_sets: tuple[np.ndarray, ...], row: int):
    width = max(value.shape[1] for value in candidate_sets)
    for rank in range(width):
        yield from (
            candidates[row, rank]
            for candidates in candidate_sets
            if rank < candidates.shape[1]
        )


def lifecycle_candidates(
    train: RetrievalQueryBatch,
    evaluation: RetrievalQueryBatch,
    corpus: RetrievalCorpus,
    top_k: int,
) -> np.ndarray:
    frequency = np.bincount(
        train.positive_item_id.numpy(), minlength=len(corpus.item_id),
    ).astype(np.float64)
    eligible = (
        corpus.active
        & surface_eligibility(int(Surface.FEED), corpus.content_kind)
    ).numpy()
    score = np.log1p(frequency) + 0.35 * corpus.quality_prior.numpy()
    groups: dict[tuple[int, int], list[int]] = {}
    topic = corpus.topic_id.numpy()
    country = corpus.country.numpy()
    for key in set(zip(topic[eligible].tolist(), country[eligible].tolist())):
        selected = np.flatnonzero(eligible & (topic == key[0]) & (country == key[1]))
        groups[key] = selected[np.argsort(-score[selected], kind="stable")].tolist()
    global_rank = np.flatnonzero(eligible)
    global_rank = global_rank[np.argsort(-score[global_rank], kind="stable")].tolist()
    output = np.full((len(evaluation.request_id), top_k), -1, dtype=np.int64)
    for row, key in enumerate(zip(
        evaluation.query_topic.tolist(), evaluation.user_country.tolist(),
    )):
        ranked = _fill_unique(groups.get(key, ()), global_rank, top_k)
        output[row, :len(ranked)] = ranked
    return output


def graph_candidates(
    train: RetrievalQueryBatch,
    evaluation: RetrievalQueryBatch,
    fallback: np.ndarray,
    top_k: int,
) -> np.ndarray:
    graph: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for source, target in zip(_last_history(train), train.positive_item_id.tolist()):
        if source >= 0 and source != target:
            graph[int(source)][int(target)] += 1
    output = np.full_like(fallback, -1)
    for row, source in enumerate(_last_history(evaluation)):
        primary = sorted(
            graph.get(int(source), {}),
            key=graph.get(int(source), {}).get,
            reverse=True,
        )
        ranked = _fill_unique(primary, fallback[row], top_k)
        output[row, :len(ranked)] = ranked
    return output


def merge_candidate_sets(
    candidate_sets: tuple[np.ndarray, ...],
    top_k: int,
) -> np.ndarray:
    if not candidate_sets or any(len(value) != len(candidate_sets[0]) for value in candidate_sets):
        raise ValueError("candidate merge requires aligned nonempty sets")
    output = np.full((len(candidate_sets[0]), top_k), -1, dtype=np.int64)
    for row in range(len(output)):
        merged = _fill_unique(_round_robin_row(candidate_sets, row), (), top_k)
        output[row, :min(top_k, len(merged))] = merged[:top_k]
    return output


def model_candidates(
    artifact: RetrievalArtifact,
    index: RetrievalANNIndex,
    batch: RetrievalQueryBatch,
    *,
    top_k: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    outputs = []
    started = perf_counter()
    artifact.model.eval()
    with torch.inference_mode():
        for start in range(0, len(batch.request_id), batch_size):
            selected = torch.arange(start, min(start + batch_size, len(batch.request_id)))
            query = artifact.model.encode_queries(batch.select(selected))
            item, _ = index.search(query, top_k)
            outputs.append(item)
    elapsed = perf_counter() - started
    return torch.cat(outputs).numpy(), elapsed * 1_000.0 / len(batch.request_id)


def ann_exact_recall(
    artifact: RetrievalArtifact,
    index: RetrievalANNIndex,
    batch: RetrievalQueryBatch,
    *,
    top_k: int,
    maximum_queries: int = 128,
) -> float:
    count = min(len(batch.request_id), maximum_queries)
    selected = batch.select(torch.arange(count))
    with torch.inference_mode():
        query = artifact.model.encode_queries(selected)
        approximate, _ = index.search(query, top_k)
        item = torch.from_numpy(index.item_embeddings).to(query.device)
        exact_rows = []
        for start in range(0, count, 16):
            state = query[start:start + 16]
            score = (
                torch.einsum("bkd,nd->bkn", state, item).max(dim=1).values
                if state.ndim == 3 else state @ item.T
            )
            location = torch.topk(score, top_k, dim=1).indices.cpu()
            exact_rows.append(index.item_ids[location])
        exact = torch.cat(exact_rows)
    overlap = torch.stack(tuple(
        torch.isin(approximate[row], exact[row]).float().mean()
        for row in range(count)
    ))
    return float(overlap.mean())


def _fixed_ranker(
    candidates: np.ndarray,
    batch: RetrievalQueryBatch,
    corpus: RetrievalCorpus,
    top_k: int,
) -> np.ndarray:
    valid = candidates >= 0
    safe = np.maximum(candidates, 0)
    topic = corpus.topic_id.numpy()[safe]
    country = corpus.country.numpy()[safe]
    quality = corpus.quality_prior.numpy()[safe]
    age = np.maximum(
        batch.event_time.numpy()[:, None] - corpus.publish_time.numpy()[safe], 0,
    )
    score = (
        1.0 * (topic == batch.query_topic.numpy()[:, None])
        + 0.25 * (country == batch.user_country.numpy()[:, None])
        + 0.35 * quality
        - 0.0002 * age
    )
    score[~valid] = -np.inf
    order = np.argsort(-score, axis=1, kind="stable")[:, :top_k]
    return np.take_along_axis(candidates, order, axis=1)


def _paired_interval(candidate_hit: np.ndarray, baseline_hit: np.ndarray) -> dict[str, float]:
    difference = candidate_hit.astype(np.float64) - baseline_hit.astype(np.float64)
    mean = float(difference.mean())
    standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
    return {
        "absolute_delta": mean,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def retrieval_metrics(
    candidates: np.ndarray,
    batch: RetrievalQueryBatch,
    corpus: RetrievalCorpus,
    *,
    downstream_k: int,
    baseline_candidates: np.ndarray | None = None,
    train_frequency: np.ndarray | None = None,
) -> dict[str, object]:
    target = batch.positive_item_id.numpy()
    hit_matrix = candidates == target[:, None]
    hit = hit_matrix.any(axis=1)
    rank = np.where(hit, hit_matrix.argmax(axis=1) + 1, 0)
    ndcg = np.zeros(len(rank), dtype=np.float64)
    ndcg[hit] = 1.0 / np.log2(rank[hit] + 1)
    valid = candidates >= 0
    downstream = _fixed_ranker(candidates, batch, corpus, downstream_k)
    downstream_hit = (downstream == target[:, None]).any(axis=1)
    metrics: dict[str, object] = {
        "recall_at_k": float(hit.mean()),
        "ndcg_at_k": float(ndcg.mean()),
        "fixed_ranker_recall_at_k": float(downstream_hit.mean()),
        "catalog_coverage": float(len(np.unique(candidates[valid])) / len(corpus.item_id)),
        "duplicate_rate": float(np.mean([
            1.0 - len(np.unique(row[row >= 0])) / max((row >= 0).sum(), 1)
            for row in candidates
        ])),
    }
    if train_frequency is not None:
        tail = train_frequency[target] <= np.quantile(
            train_frequency[train_frequency > 0], 0.5,
        )
        metrics["long_tail_query_rate"] = float(tail.mean())
        metrics["long_tail_recall_at_k"] = (
            float(hit[tail].mean()) if tail.any() else None
        )
    if baseline_candidates is not None:
        baseline_hit = (baseline_candidates == target[:, None]).any(axis=1)
        metrics["paired_recall_delta"] = _paired_interval(hit, baseline_hit)
        metrics["marginal_unique_hit_rate"] = float((hit & ~baseline_hit).mean())
        baseline_items = [set(row[row >= 0].tolist()) for row in baseline_candidates]
        metrics["marginal_unique_candidate_rate"] = float(np.mean([
            len(set(row[row >= 0].tolist()) - before) / max((row >= 0).sum(), 1)
            for row, before in zip(candidates, baseline_items)
        ]))
    return metrics


def launch_decision(
    metrics: dict[str, object],
    *,
    milliseconds_per_query: float,
    latency_budget_ms: float,
    baseline_fixed_recall: float,
) -> tuple[str, str]:
    delta = metrics["paired_recall_delta"]
    if milliseconds_per_query > latency_budget_ms:
        return "reject", "ANN latency exceeds the frozen budget"
    if metrics["fixed_ranker_recall_at_k"] < baseline_fixed_recall:
        return "reject", "fixed downstream ranker loses positive candidates"
    if delta["ci95_low"] > 0.0:
        return "pass", "paired Recall@K improves under the equal budget"
    if delta["ci95_high"] < 0.0:
        return "reject", "paired Recall@K significantly regresses"
    return "hold", "paired Recall@K interval crosses zero"
