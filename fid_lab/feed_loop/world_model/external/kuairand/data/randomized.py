"""Point-in-time KuaiRand-1K standard and randomized exposure materialization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from ..contracts import (
    DEFAULT_SEQUENCE_LENGTH,
    DENSE_NAMES,
    FEEDBACK_NAMES,
    HASH_VOCABULARIES,
    RANDOM_ITEM_POOL_SIZE,
    RANDOMIZED_SOURCE_FILES,
    RANDOMIZED_SPLIT_RATES,
    PRIOR_END_DATE,
    SPARSE_NAMES,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
)
from ..launch.contracts import stream_sha256
from .sequence import build_history


LOG_COLUMNS = (
    "user_id", "video_id", "date", "hourmin", "time_ms", *FEEDBACK_NAMES,
    "play_time_ms", "duration_ms",
)
LOG_DTYPES = {
    "user_id": "int32", "video_id": "int32", "date": "int32",
    "hourmin": "int32", "time_ms": "int64", "play_time_ms": "int32",
    "duration_ms": "int32", **{name: "uint8" for name in FEEDBACK_NAMES},
}
SPLIT_NAMES = ("train", "validation", "standard_test", "random_test")


@dataclass(frozen=True)
class RandomizedSplit:
    sparse: torch.Tensor
    dense: torch.Tensor
    history_items: torch.Tensor
    history_feedback: torch.Tensor
    labels: torch.Tensor
    user_ids: torch.Tensor
    timestamps: torch.Tensor
    dates: torch.Tensor
    raw_video_ids: torch.Tensor
    exposure_propensity: torch.Tensor

    def __len__(self) -> int:
        return len(self.labels)


def _read_log(path: Path, random: bool) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path, usecols=LOG_COLUMNS, dtype=LOG_DTYPES)
    frame["is_random_exposure"] = random
    duplicates = int(frame.duplicated(["user_id", "time_ms", "video_id"]).sum())
    if random:
        frame = frame.drop_duplicates(
            ["user_id", "time_ms", "video_id"], keep="first"
        ).reset_index(drop=True)
    return frame, duplicates


def _user_slice(frame: pd.DataFrame, user_id: int) -> pd.DataFrame:
    users = frame.user_id.to_numpy()
    start, stop = np.searchsorted(users, (user_id, user_id + 1))
    return frame.iloc[start:stop]


def _row_split(frame: pd.DataFrame) -> np.ndarray:
    dates = frame.date.to_numpy()
    random = frame.is_random_exposure.to_numpy()
    return np.select(
        (
            random,
            dates <= PRIOR_END_DATE,
            dates <= TRAIN_END_DATE,
            dates <= VALIDATION_END_DATE,
        ),
        ("random_test", "prior", "train", "validation"),
        default="standard_test",
    )


def _select_user_rows(frame, history_items, history_feedback, seed):
    assignments = _row_split(frame)
    rng = np.random.default_rng(seed + int(frame.user_id.iloc[0]))
    outputs = {}
    for name in SPLIT_NAMES:
        mask = assignments == name
        if name != "random_test":
            mask &= rng.random(len(frame)) < RANDOMIZED_SPLIT_RATES[name]
        indices = np.flatnonzero(mask)
        if len(indices):
            outputs[name] = (
                frame.iloc[indices].copy(), history_items[indices],
                history_feedback[indices],
            )
    return outputs


def _point_in_time_samples(standard_early, standard_late, random, length, seed):
    frames = defaultdict(list)
    item_histories = defaultdict(list)
    feedback_histories = defaultdict(list)
    for user_id in range(int(random.user_id.max()) + 1):
        user = pd.concat((
            _user_slice(standard_early, user_id),
            _user_slice(standard_late, user_id),
            _user_slice(random, user_id),
        ), ignore_index=True).sort_values("time_ms", kind="stable").reset_index(drop=True)
        raw_items = user.video_id.to_numpy(np.int64) % (HASH_VOCABULARIES[1] - 1) + 1
        feedback = user.loc[:, FEEDBACK_NAMES].to_numpy(np.uint8)
        history_items, history_feedback = build_history(
            user, raw_items, feedback, length
        )
        for name, values in _select_user_rows(
            user, history_items, history_feedback, seed
        ).items():
            frame, items, actions = values
            frames[name].append(frame)
            item_histories[name].append(items)
            feedback_histories[name].append(actions)
    return {
        name: (
            pd.concat(frames[name], ignore_index=True),
            np.concatenate(item_histories[name]),
            np.concatenate(feedback_histories[name]),
        )
        for name in SPLIT_NAMES
    }


def _metadata(data_dir: Path):
    item_columns = (
        "video_id", "author_id", "video_type", "upload_type", "music_type",
        "tag", "server_width", "server_height",
    )
    items = pd.read_csv(
        data_dir / RANDOMIZED_SOURCE_FILES[4], usecols=item_columns
    ).set_index("video_id")
    users = pd.read_csv(
        data_dir / RANDOMIZED_SOURCE_FILES[3]
    ).set_index("user_id")
    maps = {
        name: {value: index + 1 for index, value in enumerate(
            sorted(items[name].fillna("__missing__").astype(str).unique())
        )}
        for name in ("video_type", "upload_type")
    }
    return items, users, maps


def _encoded_features(logs, items, users, maps):
    item_rows = items.reindex(logs.video_id)
    user_rows = users.reindex(logs.user_id)
    tag = pd.to_numeric(
        item_rows.tag.astype(str).str.split(",").str[0], errors="coerce"
    ).fillna(0).to_numpy(np.int64)
    sparse = np.stack((
        logs.user_id.to_numpy(np.int64) + 1,
        logs.video_id.to_numpy(np.int64) % (HASH_VOCABULARIES[1] - 1) + 1,
        item_rows.author_id.fillna(0).to_numpy(np.int64)
        % (HASH_VOCABULARIES[2] - 1) + 1,
        tag % (HASH_VOCABULARIES[3] - 1) + 1,
        item_rows.video_type.fillna("__missing__").astype(str).map(
            maps["video_type"]
        ).to_numpy(np.int64),
        item_rows.upload_type.fillna("__missing__").astype(str).map(
            maps["upload_type"]
        ).to_numpy(np.int64),
        item_rows.music_type.fillna(0).to_numpy(np.int64)
        % (HASH_VOCABULARIES[6] - 1) + 1,
    ), axis=1)
    duration = logs.duration_ms.clip(1, 300_000).to_numpy(float)
    hour = (logs.hourmin // 100).to_numpy(float)
    width = item_rows.server_width.fillna(1).clip(1).to_numpy(float)
    height = item_rows.server_height.fillna(1).clip(1).to_numpy(float)
    dense = np.stack((
        np.log1p(duration) / np.log1p(300_000),
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.clip(width / height, 0.25, 4.0) / 4.0,
        np.log1p(user_rows.register_days.fillna(0)) / np.log1p(4_000),
        np.log1p(user_rows.follow_user_num.fillna(0)) / np.log1p(10_000),
        np.log1p(user_rows.fans_user_num.fillna(0)) / np.log1p(10_000),
        np.log1p(user_rows.friend_user_num.fillna(0)) / np.log1p(1_000),
        user_rows.is_lowactive_period.fillna(0),
        user_rows.is_live_streamer.fillna(0),
        user_rows.is_video_author.fillna(0),
    ), axis=1).astype(np.float32)
    return sparse, dense


def validate_sparse(sparse):
    for index, vocabulary in enumerate(HASH_VOCABULARIES):
        minimum, maximum = int(sparse[:, index].min()), int(sparse[:, index].max())
        if minimum < 0 or maximum >= vocabulary:
            raise ValueError(
                f"{SPARSE_NAMES[index]} range [{minimum}, {maximum}] exceeds "
                f"embedding vocabulary {vocabulary}"
            )


def _payload(logs, history_items, history_feedback, items, users, maps):
    sparse, dense = _encoded_features(logs, items, users, maps)
    validate_sparse(sparse)
    feedback = logs.loc[:, FEEDBACK_NAMES].to_numpy(np.float32)
    duration = logs.duration_ms.clip(1).to_numpy(float)
    stay = np.minimum(logs.play_time_ms.clip(0).to_numpy(float), duration)
    labels = np.concatenate((
        feedback, (np.log1p(stay) / np.log1p(duration))[:, None]
    ), axis=1).astype(np.float32)
    return {
        "sparse": torch.from_numpy(sparse),
        "dense": torch.from_numpy(dense),
        "history_items": torch.from_numpy(history_items.astype(np.int32)),
        "history_feedback": torch.from_numpy(history_feedback),
        "labels": torch.from_numpy(labels),
        "user_ids": torch.from_numpy(logs.user_id.to_numpy(np.int64, copy=True)),
        "timestamps": torch.from_numpy(logs.time_ms.to_numpy(np.int64, copy=True)),
        "dates": torch.from_numpy(logs.date.to_numpy(np.int64, copy=True)),
        "raw_video_ids": torch.from_numpy(
            logs.video_id.to_numpy(np.int64, copy=True)
        ),
        "exposure_propensity": torch.full(
            (len(logs),), 1.0 / RANDOM_ITEM_POOL_SIZE if bool(
                logs.is_random_exposure.all()
            ) else float("nan"), dtype=torch.float32,
        ),
    }


def _catalog_quality_prior(rows, standard_logs):
    prior = standard_logs[standard_logs.date <= PRIOR_END_DATE].copy()
    duration = prior.duration_ms.clip(1).to_numpy(float)
    stay = np.minimum(prior.play_time_ms.clip(0).to_numpy(float), duration)
    prior["quality_target"] = (
        0.70 * np.log1p(stay) / np.log1p(duration)
        + 0.30 * prior.long_view.to_numpy(float)
    )
    grouped = prior.groupby("video_id").quality_target.agg(["sum", "count"])
    global_mean = float(prior.quality_target.mean())
    total = rows.video_id.map(grouped["sum"]).fillna(0).to_numpy(float)
    count = rows.video_id.map(grouped["count"]).fillna(0).to_numpy(float)
    return ((total + 10.0 * global_mean) / (count + 10.0)).astype(np.float32)


def _history_topic_lookup(items):
    raw_id = items.index.to_numpy(np.int64)
    item_hash = raw_id % (HASH_VOCABULARIES[1] - 1) + 1
    tag = pd.to_numeric(
        items.tag.astype(str).str.split(",").str[0], errors="coerce"
    ).fillna(0).to_numpy(np.int64)
    lookup = np.zeros(HASH_VOCABULARIES[1], dtype=np.int64)
    lookup[item_hash] = tag % (HASH_VOCABULARIES[3] - 1) + 1
    return torch.from_numpy(lookup)


def _catalog(random_logs, standard_logs, items, users, maps):
    rows = random_logs.sort_values("time_ms").drop_duplicates("video_id")
    rows = rows.sort_values("video_id").reset_index(drop=True)
    sparse, dense = _encoded_features(rows, items, users, maps)
    standard_count = standard_logs.video_id.value_counts()
    return {
        "raw_video_ids": torch.from_numpy(rows.video_id.to_numpy(np.int64)),
        "sparse": torch.from_numpy(sparse),
        "dense": torch.from_numpy(dense),
        "standard_exposure_count": torch.from_numpy(
            rows.video_id.map(standard_count).fillna(0).to_numpy(np.int64)
        ),
        "quality_prior": torch.from_numpy(
            _catalog_quality_prior(rows, standard_logs)
        ),
        "history_topic_by_item_hash": _history_topic_lookup(items),
    }


def build_randomized_dataset(data_dir: Path, output_dir: Path,
                             source_record: str, sequence_length=DEFAULT_SEQUENCE_LENGTH,
                             seed=20260824):
    early, _ = _read_log(data_dir / RANDOMIZED_SOURCE_FILES[0], False)
    late, _ = _read_log(data_dir / RANDOMIZED_SOURCE_FILES[1], False)
    random, duplicate_rows = _read_log(
        data_dir / RANDOMIZED_SOURCE_FILES[2], True
    )
    samples = _point_in_time_samples(early, late, random, sequence_length, seed)
    items, users, maps = _metadata(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name, (logs, history_items, history_feedback) in samples.items():
        path = output_dir / f"{name}.pt"
        torch.save(
            _payload(
                logs, history_items, history_feedback, items, users, maps
            ), path,
        )
        splits[name] = {
            "rows": len(logs), "date_min": int(logs.date.min()),
            "date_max": int(logs.date.max()), "sha256": stream_sha256(path),
        }
    catalog_path = output_dir / "random_item_catalog.pt"
    torch.save(
        _catalog(random, early, items, users, maps),
        catalog_path,
    )
    manifest = {
        "schema": "kuairand-1k-randomized-sequence-v1",
        "source": source_record,
        "license": "CC-BY-4.0",
        "source_files": {
            name: stream_sha256(data_dir / name) for name in RANDOMIZED_SOURCE_FILES
        },
        "sequence_length": sequence_length,
        "prior_end_date": PRIOR_END_DATE,
        "feedback_names": FEEDBACK_NAMES,
        "sparse_names": SPARSE_NAMES,
        "sparse_vocabularies": HASH_VOCABULARIES,
        "dense_names": DENSE_NAMES,
        "sampling_rates": RANDOMIZED_SPLIT_RATES,
        "random_item_pool_size": RANDOM_ITEM_POOL_SIZE,
        "random_logging_propensity": 1.0 / RANDOM_ITEM_POOL_SIZE,
        "duplicate_random_rows_removed": duplicate_rows,
        "catalog_sha256": stream_sha256(catalog_path),
        "categorical_maps": maps,
        "splits": splits,
        "evidence_boundary": (
            "Standard exposure trains models; the untouched randomized lane is "
            "the causal-generalization and off-policy evaluation authority."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_randomized_split(dataset_dir: Path, name: str,
                          max_rows: int | None = None) -> RandomizedSplit:
    payload = torch.load(
        dataset_dir / f"{name}.pt", map_location="cpu", weights_only=False
    )
    rows = len(payload["labels"]) if max_rows is None else min(
        max_rows, len(payload["labels"])
    )
    indices = torch.arange(rows)
    return RandomizedSplit(**{
        field: payload[field][indices]
        for field in RandomizedSplit.__dataclass_fields__
    })


def calibration_masks(split: RandomizedSplit, seed: int):
    users = split.user_ids.numpy()
    labels = split.labels[:, 6].numpy()
    unique = np.unique(users)
    has_hate = np.asarray([labels[users == user].sum() > 0 for user in unique])
    calibration_users, _ = train_test_split(
        unique, test_size=0.5, random_state=seed, stratify=has_hate
    )
    calibration = np.isin(users, calibration_users)
    return calibration, ~calibration


def subset_split(split: RandomizedSplit, indices) -> RandomizedSplit:
    index = torch.as_tensor(indices, dtype=torch.long)
    return RandomizedSplit(**{
        field: getattr(split, field)[index]
        for field in RandomizedSplit.__dataclass_fields__
    })
