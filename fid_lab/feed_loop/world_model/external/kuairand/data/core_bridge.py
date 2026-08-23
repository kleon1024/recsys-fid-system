"""Bridge randomized KuaiRand requests into the core NeuralSCM contract."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from ....contracts import WORLD_LABEL_COUNT, WORLD_LABEL_NAMES
from ..launch.contracts import load_dataset_manifest, stream_sha256
from .randomized import (
    RandomizedSplit,
    calibration_masks,
    load_randomized_split,
    subset_split,
)


BRIDGE_SCHEMA = "kuairand-neural-scm-request-v1"
FEATURE_DIM = 28
SEQUENCE_LENGTH = 24


def _candidate_indices(split, catalog_size, candidates):
    offsets = torch.arange(1, candidates, dtype=torch.int64)
    seeds = (
        split.user_ids.to(torch.int64) * 1_103_515_245
        + split.timestamps.to(torch.int64) // 1_000
    )
    return torch.remainder(
        seeds[:, None] + offsets[None, :] * 2_654_435_761,
        catalog_size,
    )


def _catalog_popularity(catalog):
    counts = catalog["standard_exposure_count"].float()
    return torch.log1p(counts) / torch.log1p(counts.max().clamp_min(1.0))


def _selected_popularity(split, catalog, popularity):
    raw_ids = catalog["raw_video_ids"].long()
    positions = torch.searchsorted(raw_ids, split.raw_video_ids.long())
    valid = positions < len(raw_ids)
    safe = positions.clamp_max(len(raw_ids) - 1)
    valid &= raw_ids[safe] == split.raw_video_ids
    output = torch.zeros(len(split))
    output[valid] = popularity[safe[valid]]
    return output


def _candidate_payload(split, catalog, candidates):
    negative = _candidate_indices(split, len(catalog["sparse"]), candidates)
    sparse = torch.cat((
        split.sparse[:, None], catalog["sparse"][negative].long(),
    ), dim=1)
    dense = torch.cat((
        split.dense[:, None], catalog["dense"][negative].float(),
    ), dim=1)
    dense[:, 1:, 1:3] = split.dense[:, None, 1:3]
    dense[:, 1:, 4:] = split.dense[:, None, 4:]
    sparse[:, 1:, 0] = split.sparse[:, None, 0]
    popularity = _catalog_popularity(catalog)
    candidate_popularity = torch.cat((
        _selected_popularity(split, catalog, popularity)[:, None],
        popularity[negative],
    ), dim=1)
    return sparse, dense, candidate_popularity


def _history_matches(split, candidate_sparse):
    history = split.history_items[:, -SEQUENCE_LENGTH:].long()
    items = candidate_sparse[:, :, 1]
    matches = history[:, None, :] == items[:, :, None]
    observed = history[:, None, :] > 0
    short = (matches[:, :, -3:] & observed[:, :, -3:]).any(dim=2).float()
    long = (matches & observed).float().sum(dim=2) / observed.float().sum(
        dim=2
    ).clamp_min(1.0)
    return short, long


def _feature_tensor(split, catalog, candidates):
    sparse, dense, popularity = _candidate_payload(split, catalog, candidates)
    short, long = _history_matches(split, sparse)
    history = split.history_feedback[:, -SEQUENCE_LENGTH:].float()
    history_rows = (split.history_items[:, -SEQUENCE_LENGTH:] > 0).float()
    denominator = history_rows.sum(dim=1).clamp_min(1.0)
    engagement = (
        history[:, :, 0] + history[:, :, 1] + history[:, :, 2]
    ).sum(dim=1) / (3.0 * denominator)
    hate = history[:, :, 6].sum(dim=1) / denominator
    account_age = dense[:, :, 4]
    activity = (
        dense[:, :, 5] + dense[:, :, 6] + dense[:, :, 7]
    ) / 3.0
    lifecycle = torch.bucketize(
        account_age[:, 0].contiguous(),
        torch.tensor((math.log1p(7) / math.log1p(4_000),
                      math.log1p(30) / math.log1p(4_000),
                      math.log1p(365) / math.log1p(4_000))),
    )
    features = torch.zeros(len(split), candidates, FEATURE_DIM)
    features[:, :, 0] = (0.65 * short + 0.35 * long).clamp(0.0, 1.0)
    features[:, :, 1] = popularity.sqrt()
    features[:, :, 3] = popularity
    features[:, :, 4] = short
    features[:, :, 6] = engagement[:, None]
    features[:, :, 7] = hate[:, None]
    features[:, :, 8] = 1.0 - dense[:, :, 8]
    features[:, :, 9] = activity
    features[:, :, 10] = (denominator / SEQUENCE_LENGTH)[:, None]
    features[:, :, 11] = long
    duration_ms = torch.expm1(dense[:, :, 0] * math.log1p(300_000.0))
    features[:, :, 12] = torch.log1p(duration_ms / 1_000.0) / math.log(181.0)
    features[:, :, 14] = sparse[:, :, 0].float() / 1_001.0
    features[:, :, 15] = sparse[:, :, 1].float() / 262_144.0
    features[:, :, 16] = sparse[:, :, 2].float() / 262_144.0
    features[:, :, 17] = sparse[:, :, 3].float() / 8_192.0
    features[:, :, 19] = short
    features[:, :, 21] = 1.0
    features[:, :, 24] = account_age
    features[:, :, 25] = activity
    features[:, :, 26] = lifecycle[:, None].float() / 3.0
    features[:, :, 27] = (
        torch.remainder(split.user_ids, 10).float() / 9.0
    )[:, None]
    return features, lifecycle, popularity


def _sequence_tensor(split):
    feedback = split.history_feedback[:, -SEQUENCE_LENGTH:].float()
    items = split.history_items[:, -SEQUENCE_LENGTH:].float() / 262_144.0
    return torch.stack((
        items, torch.zeros_like(items), feedback[:, :, 1],
        torch.zeros_like(items), feedback[:, :, 2], feedback[:, :, 6],
        torch.zeros_like(items), torch.zeros_like(items),
    ), dim=2)


def _labels(split):
    duration_ms = torch.expm1(split.dense[:, 0] * math.log1p(300_000.0))
    stay_ms = torch.expm1(split.labels[:, 7] * torch.log1p(duration_ms))
    stay = stay_ms / 1_000.0
    duration = (duration_ms / 1_000.0).clamp_min(0.001)
    completion = (stay / duration).clamp(0.0, 1.0)
    labels = torch.zeros(len(split), WORLD_LABEL_COUNT)
    masks = torch.zeros_like(labels)
    labels[:, 0] = (stay > 0).float()
    labels[:, 1] = (stay >= 3).float()
    labels[:, 2] = stay
    labels[:, 3] = completion
    labels[:, 4] = (completion >= 0.95).float()
    labels[:, 5] = (stay >= torch.minimum(
        torch.full_like(stay, 18.0), duration
    )).float()
    labels[:, 6] = (stay >= torch.minimum(
        torch.full_like(stay, 30.0), duration
    )).float()
    labels[:, 7] = split.labels[:, 2]
    labels[:, 8] = split.labels[:, 6]
    labels[:, 16:20] = split.labels[:, (3, 4, 5, 0)]
    labels[:, 20] = split.labels[:, 1]
    masks[:, (0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20)] = 1.0
    return labels, masks


def bridge_split(split: RandomizedSplit, catalog, candidates=8):
    features, lifecycle, popularity = _feature_tensor(split, catalog, candidates)
    labels, masks = _labels(split)
    request_steps = (
        split.history_items[:, -SEQUENCE_LENGTH:] > 0
    ).sum(dim=1).long()
    return {
        "exposed_index": torch.zeros(len(split), dtype=torch.long),
        "exposure_propensity": torch.ones(len(split)),
        "candidate_features": features.to(torch.float16),
        "behavior_sequence": _sequence_tensor(split).to(torch.float16),
        "labels": labels,
        "label_masks": masks.to(torch.uint8),
        "lifecycle_bucket": lifecycle.to(torch.uint8),
        "region_bucket": torch.remainder(split.user_ids, 10).to(torch.uint8),
        "user_id": split.user_ids,
        "request_step": request_steps,
        "session_id": torch.zeros(len(split), dtype=torch.long),
        "candidate_fine_scores": popularity.to(torch.float16),
        "candidate_audit_utility": features[:, :, 0].to(torch.float16),
    }


def build_core_bridge(dataset_dir: Path, output_dir: Path, candidates=8):
    source = load_dataset_manifest(dataset_dir)
    catalog = torch.load(
        dataset_dir / "random_item_catalog.pt", map_location="cpu",
        weights_only=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    source_splits = {
        "train": load_randomized_split(dataset_dir, "train"),
        "validation": load_randomized_split(dataset_dir, "validation"),
    }
    randomized = load_randomized_split(dataset_dir, "random_test")
    calibration, evaluation = calibration_masks(randomized, 20260824)
    source_splits["calibration"] = subset_split(
        randomized, torch.from_numpy(calibration).nonzero().flatten()
    )
    source_splits["test"] = subset_split(
        randomized, torch.from_numpy(evaluation).nonzero().flatten()
    )
    for target_name, split in source_splits.items():
        payload = bridge_split(
            split, catalog, candidates
        )
        path = output_dir / f"{target_name}.pt"
        torch.save({"tensors": payload}, path)
        records[target_name] = {
            "source_split": (
                "random_test:user_calibration" if target_name == "calibration"
                else "random_test:user_evaluation" if target_name == "test"
                else target_name
            ),
            "rows": len(payload["labels"]),
            "sha256": stream_sha256(path),
        }
    manifest = {
        "schema": BRIDGE_SCHEMA,
        "source_manifest_sha256": sha256(
            (dataset_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_catalog_sha256": source["catalog_sha256"],
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_field_coverage": {
            "topic_norm": "video_id_hash_proxy",
            "stay_norm": "unobserved",
            "long_view": "observed_source_label",
            "quality_long_view": "unobserved",
            "like": "observed",
            "negative_feedback": "observed_hate",
            "anchor_click": "unobserved",
            "conversion": "unobserved",
        },
        "world_label_names": WORLD_LABEL_NAMES,
        "candidates_per_request": candidates,
        "splits": records,
        "evidence_boundary": (
            "External Feed actions are observed; Local, supply, retention and "
            "commercialization labels remain masked and require separate authority."
        ),
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
