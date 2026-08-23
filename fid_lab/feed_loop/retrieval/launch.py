"""Train, replay, and A/B a two-tower ANN route on one frozen Feed corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from ...evolution.data.sampling import SOURCE_FRACTIONS
from ...evolution.models.retrieval import RetrievalSnapshot, TwoTowerRetriever
from ...simulation.ab import experiment_metrics, launch_decision, randomization_audit
from ...simulation.contracts import SimulationConfig
from ...simulation.environment import build_catalog
from ...simulation.experimentation.contracts import FeedParameters
from ...simulation.policies import HeuristicPolicy
from ...simulation.population import run_population


NEGATIVES_PER_QUERY = 20


def _positive_rows(trajectories, limit: int) -> list:
    rows = [
        row
        for trajectory in trajectories
        for row in trajectory.rows
        if row.response.long_view
        or row.response.high_quality_long_view
        or row.response.anchor_click
    ]
    return rows[:limit]


def _sample_negatives(rows, catalog, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = np.asarray([row.item_id for row in rows], dtype=np.int64)
    negatives = np.empty((len(rows), NEGATIVES_PER_QUERY), dtype=np.int64)
    probabilities = np.empty_like(negatives, dtype=np.float32)
    counts = {
        source: int(NEGATIVES_PER_QUERY * fraction)
        for source, fraction in SOURCE_FRACTIONS.items()
    }
    for index, row in enumerate(rows):
        positive = row.item_id
        peers = target[target != positive]
        in_batch = rng.choice(
            peers,
            counts["in_batch"],
            replace=len(peers) < counts["in_batch"],
        )
        candidates = np.asarray(row.candidate_ids, dtype=np.int64)
        hard_pool = candidates[
            (candidates != positive)
            & (
                (catalog.category[candidates] == catalog.category[positive])
                | (catalog.city[candidates] == catalog.city[positive])
            )
        ]
        if not len(hard_pool):
            hard_pool = candidates[candidates != positive]
        hard = rng.choice(
            hard_pool,
            counts["hard"],
            replace=len(hard_pool) < counts["hard"],
        )
        random = rng.integers(len(catalog.topics) - 1, size=counts["random"])
        random += random >= positive
        negatives[index] = np.concatenate((in_batch, hard, random))
        probabilities[index] = np.concatenate(
            (
                np.full(
                    counts["in_batch"],
                    SOURCE_FRACTIONS["in_batch"] * counts["in_batch"] / len(peers),
                ),
                np.full(
                    counts["hard"],
                    SOURCE_FRACTIONS["hard"] * counts["hard"] / len(hard_pool),
                ),
                np.full(
                    counts["random"],
                    SOURCE_FRACTIONS["random"]
                    * counts["random"]
                    / (len(catalog.topics) - 1),
                ),
            )
        )
    return negatives, np.clip(probabilities, 1e-8, 1.0)


def _fit_model(rows, catalog, epochs: int, device: torch.device, seed: int):
    queries = np.asarray([row.query_embedding for row in rows], dtype=np.float32)
    targets = np.asarray([row.item_id for row in rows], dtype=np.int64)
    negatives, probabilities = _sample_negatives(rows, catalog, seed + 17)
    model = TwoTowerRetriever(queries.shape[1], catalog.topics.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 29)
    losses = []
    started = perf_counter()
    for _ in range(epochs):
        epoch_loss = []
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), 256):
            index = order[start : start + 256]
            query = torch.from_numpy(queries[index]).to(device)
            positive = torch.from_numpy(catalog.topics[targets[index]]).to(device)
            negative = torch.from_numpy(catalog.topics[negatives[index]]).to(device)
            query_state = model.encode_query(query)
            positive_score = (query_state * model.encode_item(positive)).sum(dim=1)
            negative_state = model.encode_item(
                negative.reshape(-1, catalog.topics.shape[1])
            ).reshape(len(index), NEGATIVES_PER_QUERY, -1)
            negative_score = torch.einsum("bd,bnd->bn", query_state, negative_state)
            correction = torch.from_numpy(np.log(probabilities[index])).to(device)
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
    return model, {
        "seconds": perf_counter() - started,
        "loss": losses,
        "examples": len(rows),
        "negative_source_fractions": SOURCE_FRACTIONS,
        "sampling_probability_correction": "subtract_log_q",
        "logging_policy_selection_bias": "not corrected beyond logged exploration",
    }


def _top_k(scores: torch.Tensor, top_k: int) -> np.ndarray:
    return torch.topk(scores, top_k, dim=1).indices.cpu().numpy()


def _offline_recall(model, rows, catalog, device: torch.device, top_k: int):
    queries = torch.from_numpy(
        np.asarray([row.query_embedding for row in rows], dtype=np.float32)
    ).to(device)
    targets = np.asarray([row.item_id for row in rows], dtype=np.int64)
    items = torch.from_numpy(catalog.topics).to(device)
    started = perf_counter()
    with torch.no_grad():
        baseline = _top_k(queries @ items.T, top_k)
        learned = _top_k(model.encode_query(queries) @ model.encode_item(items).T, top_k)
    latency = (perf_counter() - started) * 1_000.0 / len(rows)

    def metrics(indices):
        hit = (indices == targets[:, None]).any(axis=1)
        long_tail = catalog.popularity[targets] <= np.quantile(catalog.popularity, 0.5)
        return {
            "recall_at_k": float(hit.mean()),
            "long_tail_recall_at_k": float(hit[long_tail].mean()) if long_tail.any() else 0.0,
            "catalog_coverage": float(len(np.unique(indices)) / len(catalog.topics)),
        }

    return {
        "top_k": top_k,
        "baseline_ann": metrics(baseline),
        "trained_two_tower": {**metrics(learned), "milliseconds_per_query": latency},
    }


def _replay_delta(model, snapshot: RetrievalSnapshot, rows, catalog, device):
    audit = rows[: min(32, len(rows))]
    queries = torch.from_numpy(
        np.asarray([row.query_embedding for row in audit], dtype=np.float32)
    ).to(device)
    targets = np.asarray([row.item_id for row in audit], dtype=np.int64)
    with torch.no_grad():
        online = (
            model.encode_query(queries)
            * model.encode_item(torch.from_numpy(catalog.topics[targets]).to(device))
        ).sum(dim=1).cpu().numpy()
    replay = np.asarray(
        [snapshot.scores(row.query_embedding)[target] for row, target in zip(audit, targets)]
    )
    return float(np.max(np.abs(online - replay)))


def _fresh_user_ab(config, catalog, snapshot, users: int):
    policy = HeuristicPolicy()
    defaults = FeedParameters(fine_model=policy.name)
    treatment_parameters = defaults.overlay({"recall_model": "two_tower_trained_v2"})
    fresh_users = np.arange(users) + 90_000_000
    control = run_population(
        config, catalog, policy, fresh_users, parameters=defaults
    )
    treatment = run_population(
        config,
        catalog,
        policy,
        fresh_users,
        parameters=treatment_parameters,
        retrieval_snapshot=snapshot,
    )
    assigned = np.random.default_rng(config.seed + 404).random(users) < 0.5
    metrics, potential = experiment_metrics(control, treatment, assigned)
    return {
        "metrics": metrics,
        "decision": launch_decision(metrics),
        "randomization_audit": randomization_audit(potential, config.seed + 1404),
    }


def run_retrieval_launch(
    users: int = 1_000,
    items: int = 4_000,
    ab_users: int = 500,
    epochs: int = 8,
    top_k: int = 20,
    device: str = "cpu",
) -> dict[str, object]:
    config = SimulationConfig(users=users, items=items, joiner_users=0)
    catalog = build_catalog(config)
    logging = run_population(
        config, catalog, HeuristicPolicy(), range(users), explore=True
    )
    rows = _positive_rows(logging, limit=12_000)
    train = [row for row in rows if row.user_id % 5 != 0]
    validation = [row for row in rows if row.user_id % 5 == 0]
    if len(train) < 100 or len(validation) < 20:
        raise ValueError("insufficient mature positive rows for retrieval training")
    target_device = torch.device(device)
    torch.manual_seed(config.seed)
    model, training = _fit_model(train, catalog, epochs, target_device, config.seed)
    offline = _offline_recall(model, validation, catalog, target_device, top_k)
    corpus_hash = sha256(catalog.topics.tobytes()).hexdigest()
    version = f"two-tower-v2-{corpus_hash[:12]}"
    snapshot = model.export_snapshot(
        torch.from_numpy(catalog.topics), version
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "retrieval-snapshot.npz"
        snapshot.save(path)
        loaded = RetrievalSnapshot.load(path)
        replay_delta = _replay_delta(model, loaded, validation, catalog, target_device)
        launch = _fresh_user_ab(
            SimulationConfig(users=ab_users, items=items, joiner_users=0),
            catalog,
            loaded,
            ab_users,
        )
    offline_pass = (
        offline["trained_two_tower"]["recall_at_k"]
        >= offline["baseline_ann"]["recall_at_k"]
    )
    decision = launch["decision"] if offline_pass else "reject_offline_recall"
    return {
        "launch_id": "L-RECALL-001",
        "category": "trained_retrieval",
        "config": asdict(config),
        "split_contract": {
            "train": "positive rows from users modulo five nonzero",
            "validation": "positive rows from users modulo five zero",
            "fresh_user_ab": "disjoint user ids",
            "frozen_item_corpus": True,
        },
        "training": training,
        "offline": offline,
        "artifact": {
            "version": version,
            "item_corpus_sha256": corpus_hash,
            "shadow_replay_score_delta": replay_delta,
            "training_device": str(target_device),
            "serving_device": "cpu_numpy",
        },
        "launch": launch,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1_000)
    parser.add_argument("--items", type=int, default=4_000)
    parser.add_argument("--ab-users", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_retrieval_launch(
        args.users, args.items, args.ab_users, args.epochs, args.top_k, args.device
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
