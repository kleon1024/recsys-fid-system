"""CLI for an apples-to-apples model comparison on one FID-encoded dataset."""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import encode_rows, make_synthetic_rows
from .fid import FidCodec, FidVersion
from .models import DeepFM, ThreeTower, WideDeep
from .schema import DEFAULT_SCHEMA


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "log_loss": round(float(log_loss(labels, probabilities)), 6),
    }


def one_hot(bucket_ids: np.ndarray, bucket_sizes: list[int]) -> sparse.csr_matrix:
    offsets = np.cumsum([0, *bucket_sizes[:-1]])
    rows = np.repeat(np.arange(len(bucket_ids)), bucket_ids.shape[1])
    columns = (bucket_ids + offsets).reshape(-1)
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((values, (rows, columns)), shape=(len(bucket_ids), sum(bucket_sizes)))


def train_xgboost(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
    bucket_sizes: list[int], seed: int
) -> dict[str, float]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to run the xgboost stage") from exc
    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(one_hot(train_x, bucket_sizes), train_y)
    probabilities = model.predict_proba(one_hot(test_x, bucket_sizes))[:, 1]
    return metrics(test_y, probabilities)


def train_torch(
    model: nn.Module, train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray, epochs: int, seed: int
) -> dict[str, float]:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(test_x))).numpy()
    return metrics(test_y, probabilities)


def torch_model(name: str, bucket_sizes: list[int]) -> nn.Module:
    if name == "wide_deep":
        return WideDeep(bucket_sizes)
    if name == "deepfm":
        return DeepFM(bucket_sizes)
    if name == "three_tower":
        return ThreeTower(bucket_sizes, DEFAULT_SCHEMA)
    raise ValueError(f"unknown neural model: {name}")


def train_named_model(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    bucket_sizes: list[int],
    epochs: int,
    seed: int,
) -> dict[str, float]:
    if name == "xgboost":
        return train_xgboost(train_x, train_y, test_x, test_y, bucket_sizes, seed)
    model = torch_model(name, bucket_sizes)
    return train_torch(model, train_x, train_y, test_x, test_y, epochs, seed)


def run(models: list[str], samples: int, epochs: int, seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows, labels = make_synthetic_rows(samples, seed)
    codec = FidCodec(FidVersion.V2)
    encoded = encode_rows(rows, labels, DEFAULT_SCHEMA, codec)
    indices = np.arange(samples)
    train_indices, test_indices = train_test_split(
        indices, test_size=0.25, random_state=seed, stratify=labels
    )
    train_x, test_x = encoded.bucket_ids[train_indices], encoded.bucket_ids[test_indices]
    train_y, test_y = labels[train_indices], labels[test_indices]
    bucket_sizes = [spec.buckets for spec in DEFAULT_SCHEMA.specs]
    results: dict[str, object] = {
        "dataset": {"samples": samples, "positive_rate": round(float(labels.mean()), 6)},
        "fid": {"version": codec.version.value, "fields": len(DEFAULT_SCHEMA.specs)},
        "models": {},
    }
    for name in models:
        results["models"][name] = train_named_model(
            name, train_x, train_y, test_x, test_y, bucket_sizes, epochs, seed
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["xgboost", "wide_deep", "deepfm", "three_tower"],
        choices=["xgboost", "wide_deep", "deepfm", "three_tower"],
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args.models, args.samples, args.epochs, args.seed), indent=2))


if __name__ == "__main__":
    main()
