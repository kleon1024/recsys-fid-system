"""Bridge randomized KuaiRand requests into the core NeuralSCM contract."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from ....contracts import WORLD_LABEL_COUNT, WORLD_LABEL_NAMES
from ....feature_contract import CANONICAL_FEATURE_FIELDS, feature_contract
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
KUAI_FEATURE_CONTRACT = feature_contract(CANONICAL_FEATURE_FIELDS)
KUAI_FEATURE_COVERAGE = {
    str(index): (
        "observed_or_point_in_time_proxy"
        if index in {0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 17, 19, 20, 21, 24, 25, 26}
        else "unavailable_or_unused"
    )
    for index in range(FEATURE_DIM)
}


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


def _selected_catalog_value(split, catalog, values):
    raw_ids = catalog["raw_video_ids"].long()
    positions = torch.searchsorted(raw_ids, split.raw_video_ids.long())
    valid = positions < len(raw_ids)
    safe = positions.clamp_max(len(raw_ids) - 1)
    valid &= raw_ids[safe] == split.raw_video_ids
    output = torch.zeros(len(split))
    output[valid] = values[safe[valid]]
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
    quality = catalog["quality_prior"].float()
    candidate_quality = torch.cat((
        _selected_catalog_value(split, catalog, quality)[:, None],
        quality[negative],
    ), dim=1)
    return sparse, dense, candidate_quality


def catalog_action_payload(split, catalog, indices):
    """Build request-conditioned catalog actions without exposing outcomes."""
    sparse = catalog["sparse"][indices].long().clone()
    dense = catalog["dense"][indices].float().clone()
    dense[:, :, 1:3] = split.dense[:, None, 1:3]
    dense[:, :, 4:] = split.dense[:, None, 4:]
    sparse[:, :, 0] = split.sparse[:, None, 0]
    quality = catalog["quality_prior"].float()[indices]
    return sparse, dense, quality


def _history_matches(split, candidate_sparse, catalog):
    history_items = split.history_items[:, -SEQUENCE_LENGTH:].long()
    history = catalog["history_topic_by_item_hash"][history_items]
    topics = candidate_sparse[:, :, 3]
    matches = history[:, None, :] == topics[:, :, None]
    observed = history[:, None, :] > 0
    short = (matches[:, :, -3:] & observed[:, :, -3:]).any(dim=2).float()
    long = (matches & observed).float().sum(dim=2) / observed.float().sum(
        dim=2
    ).clamp_min(1.0)
    return short, long


def _features_from_payload(split, sparse, dense, quality, catalog):
    candidates = sparse.shape[1]
    short, long = _history_matches(split, sparse, catalog)
    history = split.history_feedback[:, -SEQUENCE_LENGTH:].float()
    history_rows = (split.history_items[:, -SEQUENCE_LENGTH:] > 0).float()
    denominator = history_rows.sum(dim=1).clamp_min(1.0)
    engagement = (
        history[:, :, 0] + history[:, :, 1] + history[:, :, 2]
    ).sum(dim=1) / (3.0 * denominator)
    hate = history[:, :, 6].sum(dim=1) / denominator
    account_age = dense[:, :, 4]
    activity = (1.0 - dense[:, :, 8]).clamp(0.0, 1.0)
    history_fill = (denominator / SEQUENCE_LENGTH).clamp(0.0, 1.0)
    lifecycle = torch.bucketize(
        account_age[:, 0].contiguous(),
        torch.tensor((math.log1p(7) / math.log1p(4_000),
                      math.log1p(30) / math.log1p(4_000),
                      math.log1p(365) / math.log1p(4_000))),
    )
    features = torch.zeros(len(split), candidates, FEATURE_DIM)
    features[:, :, 0] = (0.65 * short + 0.35 * long).clamp(0.0, 1.0)
    features[:, :, 1] = quality.sqrt()
    features[:, :, 3] = quality
    features[:, :, 4] = short
    features[:, :, 6] = engagement[:, None]
    features[:, :, 7] = hate[:, None]
    features[:, :, 8] = activity
    features[:, :, 9] = history_fill[:, None]
    features[:, :, 10] = history_fill[:, None]
    features[:, :, 11] = long
    duration_ms = torch.expm1(dense[:, :, 0] * math.log1p(300_000.0))
    features[:, :, 12] = torch.log1p(duration_ms / 1_000.0) / math.log(181.0)
    features[:, :, 17] = sparse[:, :, 3].float() / 8_192.0
    features[:, :, 19] = short
    features[:, :, 21] = 1.0
    features[:, :, 24] = account_age
    features[:, :, 25] = activity
    features[:, :, 26] = lifecycle[:, None].float() / 3.0
    return features, lifecycle, quality


def _feature_tensor(split, catalog, candidates):
    sparse, dense, quality = _candidate_payload(split, catalog, candidates)
    return _features_from_payload(split, sparse, dense, quality, catalog)


def catalog_action_features(split, catalog, indices):
    """Return NeuralSCM features for declared catalog action indices."""
    if indices.ndim != 2 or indices.shape[0] != len(split):
        raise ValueError("catalog action indices must have shape [requests, actions]")
    sparse, dense, quality = catalog_action_payload(split, catalog, indices)
    return _features_from_payload(split, sparse, dense, quality, catalog)[0]


def _sequence_tensor(split, catalog):
    feedback = split.history_feedback[:, -SEQUENCE_LENGTH:].float()
    history = split.history_items[:, -SEQUENCE_LENGTH:].long()
    topics = catalog["history_topic_by_item_hash"][history].float() / 8_192.0
    return torch.stack((
        topics, torch.zeros_like(topics), feedback[:, :, 1],
        torch.zeros_like(topics), feedback[:, :, 2], feedback[:, :, 6],
        torch.zeros_like(topics), torch.zeros_like(topics),
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
    labels[:, 5] = split.labels[:, 1]
    labels[:, 6] = (stay >= torch.minimum(
        torch.full_like(stay, 30.0), duration
    )).float()
    labels[:, 7] = split.labels[:, 2]
    labels[:, 8] = split.labels[:, 6]
    labels[:, 16:19] = split.labels[:, (3, 4, 5)]
    masks[:, (0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18)] = 1.0
    return labels, masks


def bridge_split(split: RandomizedSplit, catalog, candidates=8):
    features, lifecycle, quality = _feature_tensor(split, catalog, candidates)
    labels, masks = _labels(split)
    request_steps = (
        split.history_items[:, -SEQUENCE_LENGTH:] > 0
    ).sum(dim=1).long()
    return {
        "exposed_index": torch.zeros(len(split), dtype=torch.long),
        "exposure_propensity": torch.ones(len(split)),
        "candidate_features": features.to(torch.float16),
        "behavior_sequence": _sequence_tensor(split, catalog).to(torch.float16),
        "labels": labels,
        "label_masks": masks.to(torch.uint8),
        "lifecycle_bucket": lifecycle.to(torch.uint8),
        "region_bucket": torch.remainder(split.user_ids, 10).to(torch.uint8),
        "user_id": split.user_ids,
        "request_step": request_steps,
        "session_id": torch.zeros(len(split), dtype=torch.long),
        "event_day": split.dates.long(),
        "candidate_fine_scores": quality.to(torch.float16),
        "candidate_audit_utility": torch.full(
            (len(split), candidates), torch.nan, dtype=torch.float16
        ),
        "candidate_utility_source": "unavailable_external_randomized_bridge",
    }


def adaptation_calibration_masks(split, calibration_pool):
    """Split randomized calibration users without request-level leakage."""
    pool_users = torch.unique(split.user_ids[torch.from_numpy(calibration_pool)])
    adaptation_users = pool_users[
        torch.remainder(pool_users * 17 + 3, 5) < 3
    ]
    adaptation = calibration_pool & torch.isin(
        split.user_ids, adaptation_users
    ).numpy()
    calibration = calibration_pool & ~adaptation
    if (adaptation & calibration).any():
        raise AssertionError("randomized adaptation and calibration users overlap")
    if not adaptation.any() or not calibration.any():
        raise ValueError("randomized user split produced an empty partition")
    return adaptation, calibration


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
    calibration_pool, evaluation = calibration_masks(randomized, 20260824)
    adaptation, calibration = adaptation_calibration_masks(
        randomized, calibration_pool
    )
    source_splits["adaptation"] = subset_split(
        randomized, torch.from_numpy(adaptation).nonzero().flatten()
    )
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
                "random_test:user_adaptation" if target_name == "adaptation"
                else "random_test:user_calibration" if target_name == "calibration"
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
        "feature_contract": KUAI_FEATURE_CONTRACT,
        "feature_contract_sha256": KUAI_FEATURE_CONTRACT["sha256"],
        "feature_coverage": KUAI_FEATURE_COVERAGE,
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
            "content_click": "unobserved_in_single_column_feed",
            "source_long_view": "canonicalized_into_long_view",
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
