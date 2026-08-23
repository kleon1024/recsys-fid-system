"""Main-Feed Joiner -> online LR -> versioned PS -> shadow -> A/B loop."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ...evolution.evaluation.metrics import binary_metrics
from ...simulation.ab import experiment_metrics, launch_decision, randomization_audit
from ...simulation.contracts import SimulationConfig
from ...simulation.environment import build_catalog
from ...simulation.experiment import build_feed_joiner
from ...simulation.policies import HeuristicPolicy
from ...simulation.population import run_population
from ...training.consistency import ChainConsistencyAuditor
from ...training.contracts import ChainManifest, TrainingExample
from ...training.parameter_server import VersionedParameterServer
from ...training.trainer import OnlineMultiTaskTrainer, sigmoid


FEED_TASKS = ("long_view", "high_quality_long_view", "negative_feedback")


def _buckets(features: tuple[float, ...] | np.ndarray) -> tuple[int, ...]:
    values = np.asarray(features, dtype=float)
    return tuple(
        int(value) for value in np.clip(np.rint((values + 1.0) * 7.5), 0, 15)
    )


def joined_training_examples(fine_examples) -> list[TrainingExample]:
    examples = []
    for index, example in enumerate(fine_examples):
        if not all(example.label_masks[task] for task in FEED_TASKS):
            continue
        examples.append(
            TrainingExample(
                example_id=f"feed-online-{index}-{example.key[0]}",
                user_id=example.viewer_id,
                item_id=example.key[1],
                impression_time=index,
                feature_fids=example.feature_fids,
                feature_buckets=_buckets(example.dense_features),
                labels={task: example.labels[task] for task in FEED_TASKS},
                sample_weight=example.sample_weight,
                schema_version=example.manifest["feature"],
            )
        )
    return examples


class ParameterServerFeedPolicy:
    def __init__(
        self,
        trainer: OnlineMultiTaskTrainer,
        snapshot=None,
        name: str = "online_lr_ps",
        value_weights: tuple[float, float, float] = (1.0, 0.8, -0.3),
    ) -> None:
        self.trainer = trainer
        self.snapshot = snapshot or trainer.server.snapshot()
        self.name = name
        self.value_weights = np.asarray(value_weights)

    def _vectors(self, features: np.ndarray) -> np.ndarray:
        buckets = np.clip(np.rint((features + 1.0) * 7.5), 0, 15).astype(
            np.int64
        )
        vectors = np.zeros(
            (len(features), self.trainer.server.feature_dim), dtype=np.float64
        )
        rows = np.repeat(np.arange(len(features)), buckets.shape[1])
        fields = np.tile(np.arange(buckets.shape[1]), len(features))
        columns = (fields * 131 + buckets.reshape(-1)) % vectors.shape[1]
        np.add.at(vectors, (rows, columns), 1.0)
        return vectors

    def predict_tasks(self, features: np.ndarray) -> np.ndarray:
        vectors = self._vectors(features)
        return sigmoid(vectors @ self.snapshot.weights.T + self.snapshot.bias)

    def score(self, features: np.ndarray) -> np.ndarray:
        tasks = self.predict_tasks(features)
        return tasks @ self.value_weights


class BlendedFeedPolicy:
    def __init__(self, base, increment, weight: float = 0.25) -> None:
        self.base = base
        self.increment = increment
        self.weight = weight
        self.name = "online_lr_ps_blended"

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.base.score(features) + self.weight * self.increment.score(features)


def _train_stream(examples, epochs: int, microbatch: int):
    server = VersionedParameterServer(feature_dim=4_096, tasks=FEED_TASKS)
    trainer = OnlineMultiTaskTrainer(server, learning_rate=0.03)
    losses = []
    updates = []
    for epoch in range(epochs):
        for start in range(0, len(examples), microbatch):
            result = trainer.train_microbatch(
                examples[start : start + microbatch], f"e{epoch}-b{start}"
            )
            losses.append(result.mean_loss)
            updates.append(asdict(result.update))
    duplicate = trainer.train_microbatch(examples[:microbatch], "e0-b0").update
    return trainer, losses, updates, duplicate


def _offline_task_metrics(trainer, examples):
    scores = trainer.predict(examples)
    return {
        task: binary_metrics(
            np.asarray([example.labels[task] for example in examples]),
            scores[:, index],
        )
        for index, task in enumerate(FEED_TASKS)
    }


def evaluate_candidates(config, catalog, control_policy, candidates):
    fresh_users = np.arange(config.users) + 70_000_000
    control = run_population(config, catalog, control_policy, fresh_users)
    launches = {}
    for index, policy in enumerate(candidates):
        treatment = run_population(config, catalog, policy, fresh_users)
        assigned = np.random.default_rng(config.seed + 808 + index).random(config.users) < 0.5
        metrics, potential = experiment_metrics(control, treatment, assigned)
        launches[policy.name] = {
            "ab": metrics,
            "randomization_audit": randomization_audit(
                potential, config.seed + 1808 + index
            ),
            "decision": launch_decision(metrics),
        }
    return launches


def run_online_learning_launch(
    users: int = 1_000,
    items: int = 4_000,
    ab_users: int = 1_000,
    epochs: int = 4,
    microbatch: int = 512,
):
    config = SimulationConfig(users=users, items=items, joiner_users=users)
    catalog = build_catalog(config)
    logging_policy = HeuristicPolicy()
    logging = run_population(config, catalog, logging_policy, range(users), explore=True)
    assigned = np.zeros(users, dtype=bool)
    joined = build_feed_joiner(
        config, catalog, logging, (logging_policy, logging_policy), assigned
    )
    examples = joined_training_examples(joined.fine)
    train = [example for example in examples if example.user_id % 5 != 0]
    evaluation = [example for example in examples if example.user_id % 5 == 0]
    trainer, losses, updates, duplicate = _train_stream(train, epochs, microbatch)
    snapshot = trainer.server.snapshot()
    v1 = ParameterServerFeedPolicy(trainer, snapshot, "online_lr_ps_v1")
    balanced = ParameterServerFeedPolicy(
        trainer,
        snapshot,
        "online_lr_ps_balanced",
        (1.0, 1.5, -0.05),
    )
    candidates = (
        v1,
        balanced,
        BlendedFeedPolicy(logging_policy, balanced, 0.25),
    )
    treatment_policy = balanced
    replay_policy = ParameterServerFeedPolicy(
        trainer,
        trainer.server.snapshot(),
        treatment_policy.name,
        tuple(treatment_policy.value_weights),
    )
    audit_features = np.asarray(
        [row.features for trajectory in logging[:50] for row in trajectory.rows],
        dtype=np.float32,
    )
    replay_delta = float(
        np.max(
            np.abs(
                treatment_policy.score(audit_features)
                - replay_policy.score(audit_features)
            )
        )
    )
    evaluation_config = SimulationConfig(users=ab_users, items=items, joiner_users=0)
    evaluation_catalog = build_catalog(evaluation_config)
    launches = evaluate_candidates(
        evaluation_config,
        evaluation_catalog,
        logging_policy,
        candidates,
    )
    manifest = ChainManifest(
        examples[0].schema_version,
        "v2",
        "evolution-joiner-v1",
        snapshot.version,
        "main-feed-catalog-v1",
        FEED_TASKS,
    )
    first_scores = treatment_policy.predict_tasks(audit_features[:1])[0]
    consistency = ChainConsistencyAuditor().audit(
        manifest,
        manifest,
        examples[0],
        examples[0].feature_fids,
        first_scores,
        first_scores.copy(),
    )
    return {
        "launch_id": "L-ONLINE-001",
        "category": "realtime",
        "training": {
            "joined_examples": len(examples),
            "train_examples": len(train),
            "evaluation_examples": len(evaluation),
            "positive_rates": {
                task: float(np.mean([example.labels[task] for example in examples]))
                for task in FEED_TASKS
            },
            "epochs": epochs,
            "microbatch": microbatch,
            "mean_loss_first": float(np.mean(losses[: max(len(losses) // epochs, 1)])),
            "mean_loss_last": float(np.mean(losses[-max(len(losses) // epochs, 1) :])),
            "updates_applied": sum(update["applied"] for update in updates),
            "duplicate_update": asdict(duplicate),
            "offline_task_metrics": _offline_task_metrics(trainer, evaluation),
        },
        "parameter_server": {
            "version": snapshot.version,
            "tasks": FEED_TASKS,
        },
        "shadow_replay_score_delta": replay_delta,
        "consistency": asdict(consistency),
        "launches": launches,
        "decision": launches["online_lr_ps_blended"]["decision"],
    }
