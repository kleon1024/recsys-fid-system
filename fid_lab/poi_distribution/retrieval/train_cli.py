"""Train request-authoritative POI retrievers on the RTX 4090."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from ...feed_loop.scale.tensor_engine import TensorFeedConfig, prepare_run
from .data import load_retrieval_examples
from .features import catalog_item_features
from .models.bundle import load_bundle, save_bundle
from .models.training import train_retrievers


def _manifest(path):
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TensorFeedConfig(
        device=args.device, signal_version="kuairand-local-neural-v4"
    )
    device, _, catalog = prepare_run(config, None, 0, None)
    item_features = catalog_item_features(catalog).cpu()
    corpus_ids = torch.arange(len(item_features))
    train = load_retrieval_examples(args.dataset_dir, "train", corpus_ids)
    validation = load_retrieval_examples(
        args.dataset_dir, "validation", corpus_ids, seed=20260825
    )
    bundles, baseline = train_retrievers(
        train, validation, item_features, corpus_ids, device, args.epochs
    )
    artifacts = {}
    for name, bundle in bundles.items():
        path = args.artifact_dir / f"{name}.pt"
        artifact = save_bundle(bundle, path)
        bundle.index(item_features.to(device))
        replay = load_bundle(path, device).index(item_features.to(device))
        audit_query = validation.queries[:32].to(device)
        audit_pool = validation.negative_ids[:32, :8].to(device)
        delta = (
            bundle.score_pool(audit_query, audit_pool)
            - replay.score_pool(audit_query, audit_pool)
        ).abs().max()
        artifacts[name] = {
            "artifact": artifact,
            "offline": bundle.offline,
            "shadow_replay_max_delta": float(delta),
        }
    report = {
        "schema": "poi-retrieval-v4-training-v1",
        "dataset": _manifest(args.dataset_dir / "manifest.json"),
        "negative_sampling": {
            "in_batch": 0.60,
            "same_request_hard": 0.25,
            "random_poi": 0.15,
            "sampling_probability_correction": "subtract_log_q",
            "propensity_weighted": True,
        },
        "corpus": {
            "items": len(corpus_ids),
            "global_catalog_items": len(item_features),
            "top_k": 20,
            "same_for_all_models": True,
            "semantics": "shared Feed ANN corpus with Local behavior upweighting",
        },
        "semantic_baseline": baseline,
        "models": artifacts,
        "evidence_boundary": "Synthetic Neural Local V4 request log only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    main()
