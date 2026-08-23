"""Point-in-time KuaiRand sequence materialization with content-bound lineage."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .contracts import (
    DEFAULT_SEQUENCE_LENGTH,
    DENSE_NAMES,
    FEEDBACK_NAMES,
    SOURCE_FILES,
    SPARSE_NAMES,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
)


@dataclass(frozen=True)
class SequenceSplit:
    sparse: torch.Tensor
    dense: torch.Tensor
    history_items: torch.Tensor
    history_feedback: torch.Tensor
    labels: torch.Tensor
    user_ids: torch.Tensor
    timestamps: torch.Tensor

    def __len__(self) -> int:
        return len(self.labels)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _factorize(values: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(
        values.fillna("__missing__").astype(str), sort=True
    )
    return codes.astype(np.int64) + 1, len(uniques) + 1


def _load_logs(data_dir: Path) -> pd.DataFrame:
    columns = (
        "user_id", "video_id", "date", "hourmin", "time_ms", *FEEDBACK_NAMES,
        "play_time_ms", "duration_ms",
    )
    frames = [
        pd.read_csv(data_dir / name, usecols=columns)
        for name in SOURCE_FILES[:2]
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["user_id", "time_ms"], kind="stable"
    ).reset_index(drop=True)


def _attach_metadata(logs: pd.DataFrame, data_dir: Path):
    items = pd.read_csv(data_dir / SOURCE_FILES[3]).set_index("video_id")
    users = pd.read_csv(data_dir / SOURCE_FILES[2]).set_index("user_id")
    item_rows = items.reindex(logs.video_id)
    user_rows = users.reindex(logs.user_id)
    sparse_values = [logs.user_id.to_numpy(np.int64) + 1]
    vocabularies = [int(logs.user_id.max()) + 2]
    sparse_values.append(logs.video_id.to_numpy(np.int64) + 1)
    vocabularies.append(int(logs.video_id.max()) + 2)
    for values in (
        item_rows.author_id,
        item_rows.tag.astype(str).str.split(",").str[0],
        item_rows.video_type,
        item_rows.upload_type,
        item_rows.music_type,
    ):
        encoded, vocabulary = _factorize(values)
        sparse_values.append(encoded)
        vocabularies.append(vocabulary)
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
    return np.stack(sparse_values, axis=1), dense, tuple(vocabularies)


def _history(logs, item_ids, feedback, length):
    rows = len(logs)
    items = np.zeros((rows, length), dtype=np.int32)
    actions = np.zeros((rows, length, len(FEEDBACK_NAMES)), dtype=np.uint8)
    boundaries = np.flatnonzero(
        logs.user_id.to_numpy()[1:] != logs.user_id.to_numpy()[:-1]
    ) + 1
    for indices in np.split(np.arange(rows), boundaries):
        count = len(indices)
        for lag in range(1, min(length, count - 1) + 1):
            destination = indices[lag:]
            source = indices[:-lag]
            items[destination, length - lag] = item_ids[source]
            actions[destination, length - lag] = feedback[source]
    return items, actions


def _split_payload(payload, dates, mask):
    index = torch.from_numpy(np.flatnonzero(mask))
    return {
        name: value[index]
        for name, value in payload.items()
    } | {
        "date_min": int(dates[mask].min()),
        "date_max": int(dates[mask].max()),
    }


def build_sequence_dataset(data_dir: Path, output_dir: Path, source_commit: str,
                           sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    logs = _load_logs(data_dir)
    sparse, dense, vocabularies = _attach_metadata(logs, data_dir)
    feedback = logs.loc[:, FEEDBACK_NAMES].to_numpy(np.uint8)
    history_items, history_feedback = _history(
        logs, sparse[:, 1], feedback, sequence_length
    )
    duration = logs.duration_ms.clip(1).to_numpy(float)
    stay = np.minimum(logs.play_time_ms.clip(0).to_numpy(float), duration)
    labels = np.concatenate((
        feedback.astype(np.float32),
        (np.log1p(stay) / np.log1p(duration))[:, None].astype(np.float32),
    ), axis=1)
    payload = {
        "sparse": torch.from_numpy(sparse),
        "dense": torch.from_numpy(dense),
        "history_items": torch.from_numpy(history_items),
        "history_feedback": torch.from_numpy(history_feedback),
        "labels": torch.from_numpy(labels),
        "user_ids": torch.from_numpy(logs.user_id.to_numpy(np.int64, copy=True)),
        "timestamps": torch.from_numpy(logs.time_ms.to_numpy(np.int64, copy=True)),
    }
    dates = logs.date.to_numpy(np.int64)
    masks = {
        "train": dates <= TRAIN_END_DATE,
        "validation": (dates > TRAIN_END_DATE) & (dates <= VALIDATION_END_DATE),
        "test": dates > VALIDATION_END_DATE,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {}
    for name, mask in masks.items():
        path = output_dir / f"{name}.pt"
        split = _split_payload(payload, dates, mask)
        torch.save(split, path)
        split_manifest[name] = {
            "rows": int(mask.sum()),
            "date_min": split["date_min"],
            "date_max": split["date_max"],
            "sha256": _hash(path),
        }
    manifest = {
        "schema": "kuairand-external-sequence-v1",
        "source": "Applied-Machine-Learning-Lab/KuaiSim bundled KuaiRand-Pure",
        "source_commit": source_commit,
        "license": "CC-BY-SA-4.0",
        "source_files": {
            name: _hash(data_dir / name) for name in SOURCE_FILES
        },
        "sequence_length": sequence_length,
        "feedback_names": FEEDBACK_NAMES,
        "sparse_names": SPARSE_NAMES,
        "sparse_vocabularies": vocabularies,
        "dense_names": DENSE_NAMES,
        "splits": split_manifest,
        "evidence_boundary": (
            "External chronological behavior evidence; Pure lacks randomized logs "
            "and is not sufficient for final causal-authority promotion."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_sequence_split(dataset_dir: Path, name: str, max_rows=None) -> SequenceSplit:
    payload = torch.load(
        dataset_dir / f"{name}.pt", map_location="cpu", weights_only=False
    )
    rows = len(payload["labels"]) if max_rows is None else min(
        max_rows, len(payload["labels"])
    )
    indices = torch.arange(rows)
    return SequenceSplit(**{
        field: payload[field][indices]
        for field in SequenceSplit.__dataclass_fields__
    })
