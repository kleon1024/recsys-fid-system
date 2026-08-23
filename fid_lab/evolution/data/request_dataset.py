"""Request-level candidate authority spanning recall through mature labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ...simulation.contracts import Catalog, Trajectory
from ...value import LTMetricContainer, LTMetricVector
from .contracts import (
    TASK_WINDOWS_SECONDS,
    CandidateDecisionRecord,
    MatureLabelRecord,
    RequestCandidateDataset,
    RequestRecord,
    synthetic_impression_time,
)
from .joiner import JoinerReport


def _ranks(scores: tuple[float, ...]) -> tuple[int, ...]:
    if not scores:
        return ()
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return tuple(int(value) for value in ranks)


def _validate_trace(row) -> None:
    recall_lengths = {
        len(row.recalled_candidate_ids),
        len(row.recalled_candidate_routes),
        len(row.recall_merge_scores),
        len(row.recalled_coarse_scores),
        len(row.recalled_synthetic_oracle_scores),
        row.recall_count,
    }
    if len(recall_lengths) != 1:
        raise ValueError(f"request {row.request_id} has an open recall-stage log")
    fine_lengths = {
        len(row.candidate_ids),
        len(row.candidate_routes),
        len(row.candidate_features),
        len(row.candidate_scores),
        row.coarse_count,
    }
    if len(fine_lengths) != 1:
        raise ValueError(f"request {row.request_id} has an open coarse-stage log")
    if not set(row.candidate_ids).issubset(set(row.recalled_candidate_ids)):
        raise ValueError(f"request {row.request_id} mixes different request stages")
    if row.item_id not in row.candidate_ids:
        raise ValueError(f"request {row.request_id} exposure is outside the fine pool")


def _lt_components(row) -> Mapping[str, float]:
    if row.item_id < 0:
        return {}
    value = LTMetricContainer().evaluate(
        LTMetricVector(
            stay_minutes=row.response.watch_minutes,
            active_days=float(row.returned_next_session),
            accepted_commercialization_value=0.0,
        )
    )
    return {
        "stay": value.stay,
        "active_days": value.active_days,
        "accepted_commercialization": value.accepted_commercialization,
        "total": value.total,
    }


def _request_record(row, user_index, history, manifest) -> RequestRecord:
    return RequestRecord(
        row.request_id,
        row.user_id,
        row.session_id,
        row.request_index,
        synthetic_impression_time(user_index, row.session_id, row.request_index),
        dict(row.parameter_snapshot or {}),
        dict(manifest),
        tuple(history[-24:]),
    )


def _candidate_record(row, catalog, recall_index, fine_index, coarse_rank, fine_rank):
    candidate_id = row.recalled_candidate_ids[recall_index]
    index = fine_index.get(candidate_id)
    coarse_pass = index is not None
    exposed = candidate_id == row.item_id
    current_fine_rank = fine_rank[index] if index is not None else None
    filter_reason = None if exposed else (
        "fine_ranked_out" if coarse_pass else "coarse_filtered"
    )
    fine_score = row.candidate_scores[index] if index is not None else None
    return CandidateDecisionRecord(
        row.request_id,
        candidate_id,
        int(catalog.author[candidate_id]),
        int(catalog.poi[candidate_id]),
        row.recalled_candidate_routes[recall_index],
        row.recall_merge_scores[recall_index],
        recall_index + 1,
        row.recalled_synthetic_oracle_scores[recall_index],
        candidate_id == row.corpus_oracle_item_id,
        row.recalled_coarse_scores[recall_index],
        coarse_rank[recall_index],
        coarse_pass,
        row.candidate_features[index] if index is not None else (),
        fine_score,
        current_fine_rank,
        fine_score,
        fine_score,
        current_fine_rank,
        1 if exposed else None,
        filter_reason,
    )


def _label_record(row, catalog, candidate, label_source) -> MatureLabelRecord:
    key = (row.request_id, candidate.candidate_id, candidate.poi_id)
    source = label_source.get(key)
    empty_labels = {task: 0.0 for task in TASK_WINDOWS_SECONDS}
    empty_masks = {task: False for task in TASK_WINDOWS_SECONDS}
    exposed = candidate.candidate_id == row.item_id
    return MatureLabelRecord(
        row.request_id,
        candidate.candidate_id,
        int(catalog.poi[candidate.candidate_id]),
        source.hard_labels if source is not None else empty_labels,
        source.label_masks if source is not None else empty_masks,
        _lt_components(row) if exposed and source is not None else {},
    )


def _row_records(row, catalog, label_source):
    coarse_rank = _ranks(row.recalled_coarse_scores)
    fine_rank = _ranks(row.candidate_scores)
    fine_index = {
        candidate_id: index for index, candidate_id in enumerate(row.candidate_ids)
    }
    candidates = tuple(
        _candidate_record(
            row, catalog, index, fine_index, coarse_rank, fine_rank
        )
        for index in range(row.recall_count)
    )
    labels = tuple(
        _label_record(row, catalog, candidate, label_source)
        for candidate in candidates
    )
    return candidates, labels


def _attribution(row, records) -> str:
    recalled = {record.candidate_id for record in records}
    coarse = {record.candidate_id for record in records if record.coarse_pass}
    oracle = row.corpus_oracle_item_id
    if oracle not in recalled:
        return "recall_miss"
    if oracle not in coarse:
        return "coarse_miss"
    if oracle == row.item_id:
        return "served_oracle"
    oracle_record = next(record for record in records if record.candidate_id == oracle)
    return (
        "mix_rank_miss"
        if oracle_record.mix_rank != oracle_record.fine_rank
        else "fine_rank_miss"
    )


def build_request_candidate_dataset(
    trajectories: list[Trajectory],
    catalog: Catalog,
    joiner_report: JoinerReport,
    manifest: Mapping[str, str],
) -> RequestCandidateDataset:
    """Close request, stage decisions, PIT history, and labels without false negatives."""
    label_source = {example.key: example for example in joiner_report.coarse}
    requests: list[RequestRecord] = []
    candidates: list[CandidateDecisionRecord] = []
    labels: list[MatureLabelRecord] = []
    attribution: Counter[str] = Counter({
        name: 0 for name in (
            "recall_miss", "coarse_miss", "fine_rank_miss",
            "mix_rank_miss", "served_oracle",
        )
    })
    for user_index, trajectory in enumerate(trajectories):
        history: list[tuple[float, ...]] = []
        for row in trajectory.rows:
            _validate_trace(row)
            requests.append(_request_record(row, user_index, history, manifest))
            row_candidates, row_labels = _row_records(row, catalog, label_source)
            candidates.extend(row_candidates)
            labels.extend(row_labels)
            attribution[_attribution(row, row_candidates)] += 1
            history.append(row.features[:8])
    attribution["requests"] = len(requests)
    return RequestCandidateDataset(
        tuple(requests), tuple(candidates), tuple(labels), dict(attribution)
    )


def dataset_tables(dataset: RequestCandidateDataset) -> dict[str, list[dict[str, object]]]:
    """Physical-table view for Parquet, ClickHouse, or warehouse materialization."""
    return {
        "requests": [asdict(value) for value in dataset.requests],
        "candidate_decisions": [asdict(value) for value in dataset.candidates],
        "mature_labels": [asdict(value) for value in dataset.labels],
    }


def materialize_dataset(
    dataset: RequestCandidateDataset,
    output_dir: Path,
) -> dict[str, object]:
    """Write ClickHouse-compatible JSONL partitions plus a content manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "request-candidate-v1",
        "stage_attribution": dict(dataset.stage_attribution),
        "tables": {},
    }
    for name, rows in dataset_tables(dataset).items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        manifest["tables"][name] = {
            "rows": len(rows),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
