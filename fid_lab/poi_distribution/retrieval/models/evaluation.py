"""Equal-corpus offline metrics and fixed-pool serving latency."""

from __future__ import annotations

from time import perf_counter

import torch


def _metrics(indices, target, poi_ids, high_value):
    retrieved = poi_ids[indices]
    hit = (retrieved == target[:, None]).any(dim=1)
    rank = (retrieved == target[:, None]).float().argmax(dim=1) + 1
    ndcg = torch.where(
        hit, 1.0 / torch.log2(rank.float() + 1.0), torch.zeros_like(rank).float()
    )
    return {
        "recall_at_k": float(hit.float().mean()),
        "ndcg_at_k": float(ndcg.mean()),
        "high_value_recall_at_k": (
            float(hit[high_value].float().mean()) if high_value.any() else None
        ),
        "catalog_coverage": float(torch.unique(retrieved).numel() / len(poi_ids)),
    }


def _model_topk(bundle, query, top_k, chunk=512):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(query), chunk):
            encoded = bundle.encode_query(query[start:start + chunk])
            if encoded.ndim == 3:
                scores = torch.einsum(
                    "bid,nd->bin", encoded, bundle.item_embeddings
                ).max(dim=1).values
            else:
                scores = encoded @ bundle.item_embeddings.T
            rows.append(torch.topk(scores, top_k, dim=1).indices.cpu())
    return torch.cat(rows)


def evaluate_bundle(bundle, examples, item_features, poi_ids, top_k=20):
    device = bundle.query_mean.device
    bundle.index(item_features)
    query = examples.queries.to(device)
    started = perf_counter()
    indices = _model_topk(bundle, query, top_k)
    exact_ms = (perf_counter() - started) * 1_000.0 / len(query)
    audit_rows = min(len(query), 8_192)
    generator = torch.Generator(device=device).manual_seed(20260824)
    pools = torch.randint(
        len(item_features), (audit_rows, 24), generator=generator, device=device
    )
    started = perf_counter()
    with torch.inference_mode():
        bundle.score_pool(query[:audit_rows], pools)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    route_ms = (perf_counter() - started) * 1_000.0 / audit_rows
    result = _metrics(
        indices, examples.positive_ids, poi_ids.cpu(), examples.high_value
    )
    result.update({
        "exact_full_corpus_ms_per_query": exact_ms,
        "fixed_pool_ms_per_query": route_ms,
        "corpus_items": len(item_features),
        "top_k": top_k,
    })
    return result


def semantic_baseline(examples, item_features, poi_ids, top_k=20):
    query = examples.queries[:, :12]
    item = item_features[:, :12].cpu()
    rows = []
    for start in range(0, len(query), 1_024):
        rows.append(torch.topk(query[start:start + 1_024] @ item.T, top_k, dim=1).indices)
    return _metrics(
        torch.cat(rows), examples.positive_ids, poi_ids.cpu(), examples.high_value
    )
