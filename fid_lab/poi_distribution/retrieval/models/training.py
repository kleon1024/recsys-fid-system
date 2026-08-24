"""IPS-weighted sampled-softmax training for POI retrieval models."""

from __future__ import annotations

from time import perf_counter

import torch

from .architectures import build_retriever
from .bundle import RetrievalBundle, corpus_hash
from .evaluation import evaluate_bundle, semantic_baseline


def _batch_loss(model, query, positive, negative, probability, weights, temperature):
    query_state = model.encode_query(query)
    positive_state = model.encode_item(positive)
    negative_state = model.encode_item(negative.flatten(0, 1)).reshape(
        len(query), negative.shape[1], -1
    )
    if query_state.ndim == 3:
        positive_score = torch.einsum(
            "bid,bd->bi", query_state, positive_state
        ).max(dim=1).values
        negative_score = torch.einsum(
            "bid,bnd->bin", query_state, negative_state
        ).max(dim=1).values
    else:
        positive_score = (query_state * positive_state).sum(dim=1)
        negative_score = torch.einsum("bd,bnd->bn", query_state, negative_state)
    logits = torch.cat((
        positive_score[:, None] / temperature,
        negative_score / temperature - probability.log(),
    ), dim=1)
    point = torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(query), dtype=torch.long, device=query.device),
        reduction="none",
    )
    return (point * weights).mean()


def _fit(name, train, validation, item_features, corpus_ids, device, epochs, seed):
    torch.manual_seed(seed)
    model = build_retriever(name).to(device)
    query_mean = train.queries.mean(0).to(device)
    query_scale = train.queries.std(0).clamp_min(1e-4).to(device)
    corpus_features = item_features[corpus_ids]
    item_mean = corpus_features.mean(0).to(device)
    item_scale = corpus_features.std(0).clamp_min(1e-4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 19)
    losses = []
    started = perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(train), generator=generator)
        epoch = []
        for start in range(0, len(order), 1_024):
            index = order[start:start + 1_024]
            query = (train.queries[index].to(device) - query_mean) / query_scale
            positive = (
                item_features[train.positive_ids[index]].to(device) - item_mean
            ) / item_scale
            negative = (
                item_features[train.negative_ids[index]].to(device) - item_mean
            ) / item_scale
            loss = _batch_loss(
                model, query, positive, negative,
                train.negative_probability[index].to(device),
                train.weights[index].to(device), 0.08,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch.append(float(loss.detach()))
        losses.append(sum(epoch) / len(epoch))
    bundle = RetrievalBundle(
        name, model, query_mean, query_scale, item_mean, item_scale,
        corpus_hash(corpus_features), {},
    )
    offline = evaluate_bundle(
        bundle, validation, corpus_features.to(device), corpus_ids.to(device)
    )
    bundle.offline = {
        **offline,
        "loss": losses,
        "training_seconds": perf_counter() - started,
        "parameters": sum(value.numel() for value in model.parameters()),
        "train_examples": len(train),
        "validation_examples": len(validation),
    }
    return bundle


def train_retrievers(train, validation, item_features, corpus_ids, device, epochs=4):
    baseline = semantic_baseline(
        validation, item_features[corpus_ids], corpus_ids
    )
    bundles = {
        name: _fit(
            name, train, validation, item_features, corpus_ids,
            torch.device(device), epochs, 20260824 + offset,
        )
        for offset, name in enumerate(("two_tower", "multi_interest"))
    }
    return bundles, baseline
