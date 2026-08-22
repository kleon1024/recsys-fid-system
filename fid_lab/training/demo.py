"""Executable impression-to-online-model example with AUC comparison."""

from __future__ import annotations

import json

import numpy as np

from .consistency import ChainConsistencyAuditor
from .contracts import ActionEvent, ChainManifest, ImpressionEvent, PredictionRecord, TASKS
from .evaluation import compare
from .joiner import ExampleJoiner
from .parameter_server import VersionedParameterServer
from .trainer import OnlineMultiTaskTrainer


def make_events(size: int = 600, seed: int = 23) -> tuple[list[ImpressionEvent], list[ActionEvent]]:
    rng = np.random.default_rng(seed)
    impressions: list[ImpressionEvent] = []
    actions: list[ActionEvent] = []
    for index in range(size):
        buckets = tuple(int(value) for value in rng.integers(0, 64, size=9))
        event_time = index * 10
        impression = ImpressionEvent(
            request_id=f"r-{index}",
            user_id=index % 50,
            item_id=index,
            event_time=event_time,
            position=index % 20,
            propensity=max(0.08, 1.0 / (1.0 + index % 20)),
            feature_fids=tuple((field + 1) << 48 | bucket for field, bucket in enumerate(buckets)),
            feature_buckets=buckets,
            schema_version="fid-schema-v1",
            served_model_version=0,
        )
        impressions.append(impression)
        logits = np.asarray(
            [
                1.2 * (buckets[0] % 3 == 0) - 0.8,
                1.0 * (buckets[3] % 4 == 0) - 1.1,
                1.3 * (buckets[5] % 5 <= 1) - 0.9,
            ]
        )
        outcomes = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
        for task, outcome in zip(TASKS, outcomes):
            if outcome:
                actions.append(
                    ActionEvent(
                        event_id=f"a-{index}-{task}",
                        request_id=impression.request_id,
                        item_id=index,
                        action=task,
                        event_time=event_time + 20,
                        received_at=event_time + 25,
                    )
                )
    return impressions, actions


def records(examples: list, scores: np.ndarray, task_index: int) -> list[PredictionRecord]:
    task = TASKS[task_index]
    return [
        PredictionRecord(
            example.user_id,
            example.labels[task],
            float(scores[index, task_index]),
            0,
            "head" if example.item_id % 5 else "tail",
        )
        for index, example in enumerate(examples)
    ]


def main() -> None:
    impressions, actions = make_events()
    joiner = ExampleJoiner()
    watermark = impressions[-1].event_time + 1000
    joined = joiner.build(impressions, actions, watermark)
    examples = list(joined.examples)
    split = int(len(examples) * 0.7)
    train, evaluation = examples[:split], examples[split:]
    server = VersionedParameterServer()
    trainer = OnlineMultiTaskTrainer(server)
    for epoch in range(6):
        for start in range(0, len(train), 64):
            trainer.train_microbatch(train[start : start + 64], f"epoch-{epoch}-batch-{start}")
    scores = trainer.predict(evaluation)
    offline = records(evaluation, scores, 0)
    online = [
        PredictionRecord(
            record.user_id,
            record.label,
            min(1.0, max(0.0, record.score * 0.92 + 0.03)),
            server.snapshot().version,
            record.slice_name,
        )
        for record in offline
    ]
    metric_report = compare(offline, online)
    manifest = ChainManifest(
        "fid-schema-v1", "v2", joiner.config.version, server.snapshot().version, "viking-local-v1"
    )
    audit = ChainConsistencyAuditor().audit(
        manifest,
        manifest,
        evaluation[0],
        evaluation[0].feature_fids,
        scores[0],
        scores[0].copy(),
    )
    print(
        json.dumps(
            {
                "events": {"impressions": len(impressions), "actions": len(actions)},
                "joiner": {
                    "examples": len(examples),
                    "immature": joined.immature_impressions,
                    "ignored_actions": joined.ignored_actions,
                },
                "parameter_server": {"model_version": server.snapshot().version},
                "click_metrics": {
                    "offline_auc": metric_report.offline.auc,
                    "online_auc": metric_report.online.auc,
                    "auc_gap": metric_report.auc_gap,
                    "online_calibration_error": metric_report.online.calibration_error,
                },
                "consistency": {"passed": audit.passed, "checks": audit.checks},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
