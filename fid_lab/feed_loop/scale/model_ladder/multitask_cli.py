"""Train only the stay-aligned primitive multi-task challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ...models.artifact import publish_policy
from ...models.feed_multitask import FeedMultiTaskPolicy
from .data import exposed_split, load_tensors
from .evaluate import candidate_metrics, multitask_metrics, offline_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train = exposed_split(args.dataset_dir, "train")
    validation = exposed_split(args.dataset_dir, "validation")
    test = exposed_split(args.dataset_dir, "test")
    probability = train.weights / train.weights.sum()
    indices = np.random.default_rng(20260823).choice(
        len(train.features), len(train.features), p=probability
    )
    model = FeedMultiTaskPolicy(train.features.shape[1], args.device)
    started = perf_counter()
    model.fit(
        train.features[indices], train.labels[indices],
        validation.features, validation.labels, args.epochs,
    )
    training_seconds = perf_counter() - started
    published, replay = publish_policy(
        model, validation.features[:20_000], "kuairand-calibrated-v3",
        args.artifact_dir,
    )
    evidence = {
        "offline": offline_metrics(model, test),
        "primitive_heads": multitask_metrics(model, test),
        "candidate": candidate_metrics(
            model, load_tensors(args.dataset_dir, "test")
        ),
        "training_seconds": training_seconds,
        "parameters": model.parameters,
        "loss_history": model.loss_history,
        "shadow_replay_max_delta": replay,
        "artifact_manifest": dict(published.artifact_manifest),
    }
    report = {
        "schema": "v3-stay-aligned-multitask-training-v1",
        "dataset_manifest": json.loads(
            (args.dataset_dir / "manifest.json").read_text()
        ),
        "target_contract": "primitive heads; linear stay; LT evaluation only",
        "models": {model.name: evidence},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({model.name: evidence}, indent=2))


if __name__ == "__main__":
    main()
