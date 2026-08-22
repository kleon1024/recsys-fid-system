"""Deterministic synthetic CTR data and one shared encoding path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fid import FidCodec
from .schema import FeatureRegistry


@dataclass(frozen=True)
class EncodedDataset:
    bucket_ids: np.ndarray
    fids: np.ndarray
    labels: np.ndarray


def make_synthetic_rows(n: int = 5000, seed: int = 7) -> tuple[list[dict[str, object]], np.ndarray]:
    rng = np.random.default_rng(seed)
    user_affinity = rng.normal(0, 0.7, 240)
    item_quality = rng.normal(0, 0.65, 360)
    category_affinity = rng.normal(0, 0.5, (6, 12))

    rows: list[dict[str, object]] = []
    logits = np.empty(n, dtype=np.float32)
    for index in range(n):
        user_id = int(rng.integers(240))
        item_id = int(rng.integers(360))
        age_bucket = int(user_id % 6)
        category = int(item_id % 12)
        country = int((user_id * 7) % 6)
        device = int(rng.integers(3))
        hour_bucket = int(rng.integers(4))
        row = {
            "user_id": user_id,
            "age_bucket": age_bucket,
            "item_id": item_id,
            "category": category,
            "country": country,
            "device": device,
            "hour_bucket": hour_bucket,
        }
        rows.append(row)
        logits[index] = (
            -1.0
            + user_affinity[user_id]
            + item_quality[item_id]
            + category_affinity[country, category]
            + 0.65 * (category % 3 == device)
            + 0.35 * (age_bucket % 4 == hour_bucket)
        )

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = rng.binomial(1, probabilities).astype(np.float32)
    return rows, labels


def encode_rows(
    rows: list[dict[str, object]], labels: np.ndarray, registry: FeatureRegistry, codec: FidCodec
) -> EncodedDataset:
    encoded = [registry.encode_row(row, codec) for row in rows]
    fids = np.asarray([item[0] for item in encoded], dtype=np.uint64)
    bucket_ids = np.asarray([item[1] for item in encoded], dtype=np.int64)
    return EncodedDataset(bucket_ids=bucket_ids, fids=fids, labels=labels)
