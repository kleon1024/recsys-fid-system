"""Train and evaluate a fixed-corpus external Feed retrieval ladder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from ..contracts import HASH_VOCABULARIES
from ..data.randomized import (
    calibration_masks,
    load_randomized_split,
    subset_split,
)
from ..launch.contracts import load_dataset_manifest, stream_sha256
from .model import KuaiMultiInterestRetriever, KuaiTwoTowerRetriever


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 50
    epochs: int = 6
    batch_size: int = 1_024
    hard_negatives: int = 5
    random_negatives: int = 3
    learning_rate: float = 2e-3
    seed: int = 20260824
    evaluation_seed: int = 20260824


def _catalog_index(raw_video_ids, catalog_raw_ids):
    positions = torch.searchsorted(catalog_raw_ids, raw_video_ids)
    valid = positions < len(catalog_raw_ids)
    safe = positions.clamp_max(len(catalog_raw_ids) - 1)
    valid &= catalog_raw_ids[safe] == raw_video_ids
    return safe, valid


def _positive_rows(split, catalog):
    indices, in_catalog = _catalog_index(
        split.raw_video_ids.long(), catalog["raw_video_ids"].long()
    )
    strength = split.labels[:, :6].sum(dim=1)
    return torch.nonzero(in_catalog & (strength > 0)).flatten(), indices


def _negative_table(catalog, config):
    rng = np.random.default_rng(config.seed + 31)
    tags = catalog["sparse"][:, 3].numpy()
    rows = []
    probabilities = []
    for index, tag in enumerate(tags):
        pool = np.flatnonzero((tags == tag) & (np.arange(len(tags)) != index))
        if not len(pool):
            pool = np.flatnonzero(np.arange(len(tags)) != index)
        hard = rng.choice(
            pool, config.hard_negatives,
            replace=len(pool) < config.hard_negatives,
        )
        random_pool = np.flatnonzero(np.arange(len(tags)) != index)
        random = rng.choice(
            random_pool, config.random_negatives, replace=False
        )
        rows.append(np.concatenate((hard, random)))
        probabilities.append(np.concatenate((
            np.full(config.hard_negatives, config.hard_negatives / len(pool)),
            np.full(
                config.random_negatives,
                config.random_negatives / len(random_pool),
            ),
        )))
    return (
        torch.from_numpy(np.asarray(rows, dtype=np.int64)),
        torch.from_numpy(np.asarray(probabilities, dtype=np.float32)),
    )


def _scores(query, items):
    if query.ndim == 3:
        return torch.einsum("bkd,bnd->bkn", query, items).max(dim=1).values
    return torch.einsum("bd,bnd->bn", query, items)


def _training_loss(model, batch, positive_index, catalog, negative_table,
                   negative_probability, device):
    query = model.encode_query(
        batch.sparse.to(device), batch.dense.to(device),
        batch.history_items.to(device), batch.history_feedback.to(device),
    )
    item_sparse = catalog["sparse"][positive_index].to(device)
    item_dense = catalog["dense"][positive_index].to(device)
    positive_state = model.encode_item(item_sparse, item_dense)
    positive = _scores(query, positive_state[:, None])
    in_batch_items = positive_state[None].expand(len(positive_index), -1, -1)
    in_batch = _scores(query, in_batch_items)
    duplicate = positive_index[:, None] == positive_index[None]
    in_batch = in_batch.masked_fill(duplicate.to(device), -torch.inf)
    sampled_index = negative_table[positive_index]
    sampled = model.encode_item(
        catalog["sparse"][sampled_index].reshape(-1, 7).to(device),
        catalog["dense"][sampled_index].reshape(-1, 11).to(device),
    ).reshape(len(positive_index), sampled_index.shape[1], -1)
    sampled_score = _scores(query, sampled)
    correction = torch.log(
        negative_probability[positive_index].to(device).clamp_min(1e-8)
    )
    logits = torch.cat((
        positive / 0.08,
        in_batch / 0.08,
        sampled_score / 0.08 - correction,
    ), dim=1)
    return torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(logits), dtype=torch.long, device=device)
    )


def _batch(split, indices):
    return subset_split(split, indices)


def train_retriever(model, split, catalog, config, device, max_rows=None):
    positive_rows, catalog_index = _positive_rows(split, catalog)
    if max_rows is not None:
        positive_rows = positive_rows[:max_rows]
    negative_table, negative_probability = _negative_table(catalog, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    started = perf_counter()
    model.to(device)
    for epoch in range(config.epochs):
        model.train()
        order = positive_rows[torch.randperm(len(positive_rows), generator=generator)]
        losses = []
        for start in range(0, len(order), config.batch_size):
            rows = order[start:start + config.batch_size]
            loss = _training_loss(
                model, _batch(split, rows), catalog_index[rows], catalog,
                negative_table, negative_probability, device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return {
        "positive_examples": len(positive_rows),
        "history": history,
        "seconds": perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "negative_sampling": {
            "in_batch": "all non-duplicate positives",
            "hard_same_tag": config.hard_negatives,
            "random_catalog": config.random_negatives,
            "sampled_log_q_correction": True,
        },
    }


@torch.inference_mode()
def _model_candidates(model, split, catalog, rows, config, device):
    model.eval()
    item_state = model.encode_item(
        catalog["sparse"].to(device), catalog["dense"].to(device)
    )
    outputs = []
    for start in range(0, len(rows), config.batch_size):
        index = rows[start:start + config.batch_size]
        batch = _batch(split, index)
        query = model.encode_query(
            batch.sparse.to(device), batch.dense.to(device),
            batch.history_items.to(device), batch.history_feedback.to(device),
        )
        score = (
            torch.einsum("bkd,nd->bkn", query, item_state).max(dim=1).values
            if query.ndim == 3 else query @ item_state.T
        )
        outputs.append(torch.topk(score, config.top_k, dim=1).indices.cpu())
    return torch.cat(outputs)


def _popular_candidates(catalog, rows, top_k):
    popular = torch.topk(catalog["standard_exposure_count"], top_k).indices
    return popular[None].expand(len(rows), -1)


def _graph_candidates(train, test, catalog, test_rows, top_k):
    train_rows, train_targets = _positive_rows(train, catalog)
    graph = {}
    for row in train_rows.tolist():
        history = train.history_items[row]
        observed = history[history > 0]
        if not len(observed):
            continue
        key = int(observed[-1])
        target = int(train_targets[row])
        graph.setdefault(key, {})[target] = graph.setdefault(key, {}).get(target, 0) + 1
    fallback = _popular_candidates(catalog, test_rows, top_k)[0].tolist()
    output = []
    for row in test_rows.tolist():
        history = test.history_items[row]
        observed = history[history > 0]
        key = int(observed[-1]) if len(observed) else -1
        ranked = sorted(graph.get(key, {}), key=graph.get(key, {}).get, reverse=True)
        ranked.extend(item for item in fallback if item not in ranked)
        output.append(ranked[:top_k])
    return torch.tensor(output)


def _metrics(candidates, target, popularity):
    hit = candidates == target[:, None]
    found = hit.any(dim=1)
    rank = torch.where(found, hit.float().argmax(dim=1) + 1, 0)
    ndcg = torch.zeros(len(target))
    ndcg[found] = 1.0 / torch.log2(rank[found].float() + 1.0)
    tail = popularity[target] <= torch.quantile(popularity.float(), 0.5)
    return {
        "recall_at_k": float(found.float().mean()),
        "ndcg_at_k": float(ndcg.mean()),
        "long_tail_recall_at_k": float(found[tail].float().mean()),
        "catalog_coverage": float(candidates.unique().numel() / len(popularity)),
    }


def _artifact(model, path, manifest, config, name):
    torch.save({
        "schema": "kuairand-retrieval-artifact-v1", "name": name,
        "config": asdict(config), "dataset_manifest": manifest,
        "state_dict": model.state_dict(),
    }, path)
    return {"path": path.name, "sha256": stream_sha256(path)}


def run_retrieval_ladder(dataset_dir, output_dir, config, device="cuda:0",
                         max_train_rows=None):
    torch.manual_seed(config.seed)
    manifest = load_dataset_manifest(dataset_dir)
    catalog = torch.load(
        dataset_dir / "random_item_catalog.pt", map_location="cpu",
        weights_only=False,
    )
    train = load_randomized_split(dataset_dir, "train")
    randomized = load_randomized_split(dataset_dir, "random_test")
    _, evaluation = calibration_masks(randomized, config.evaluation_seed)
    test = subset_split(randomized, np.flatnonzero(evaluation))
    test_rows, target = _positive_rows(test, catalog)
    target = target[test_rows]
    models = {
        "two_tower": KuaiTwoTowerRetriever(HASH_VOCABULARIES),
        "multi_interest": KuaiMultiInterestRetriever(HASH_VOCABULARIES),
    }
    candidates = {
        "popular": _popular_candidates(catalog, test_rows, config.top_k),
        "co_visit_graph": _graph_candidates(
            train, test, catalog, test_rows, config.top_k
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    training = {}
    artifacts = {}
    for name, model in models.items():
        training[name] = train_retriever(
            model, train, catalog, config, torch.device(device), max_train_rows
        )
        candidates[name] = _model_candidates(
            model, test, catalog, test_rows, config, torch.device(device)
        )
        artifacts[name] = _artifact(
            model, output_dir / f"{name}.pt", manifest, config, name
        )
    popularity = catalog["standard_exposure_count"]
    return {
        "schema": "kuairand-randomized-retrieval-ladder-v1",
        "config": asdict(config),
        "dataset_catalog_sha256": manifest["catalog_sha256"],
        "evaluation": {
            "randomized_users": int(test.user_ids[test_rows].unique().numel()),
            "positive_queries": len(test_rows),
            "user_disjoint_calibration_evaluation": True,
            "same_corpus_top_k": True,
        },
        "models": {
            name: {**_metrics(value, target, popularity),
                   **({"training": training[name]} if name in training else {})}
            for name, value in candidates.items()
        },
        "artifacts": artifacts,
        "candidate_sets": candidates,
        "test_rows": test_rows,
        "target": target,
        "evidence_boundary": (
            "Random-exposure offline retrieval evidence; downstream fixed-rank "
            "shadow and simulated A/B remain required for launch."
        ),
    }
