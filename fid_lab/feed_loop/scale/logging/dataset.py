"""Generate a frozen request-level training snapshot from the V3 candidate graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch

from ...authority import attach_dataset
from ...tensor_cascade import (
    _fine_score,
    coarse_rank,
    materialize_selected,
)
from ...tensor_policies import PERSONALIZED
from ..artifact.features import build_tensor_features
from ..experiment.trigger import refresh_search_state
from ..graph.random import uniform
from ..tensor_engine import (
    TensorFeedConfig,
    candidate_batch,
    prepare_run,
    sample_step,
)
from ..tensor_runtime.state import advance_state, new_user_state


LABEL_NAMES = (
    "play", "play_3s", "stay_seconds", "play_completion_ratio",
    "complete_play", "long_view", "quality_long_view", "like",
    "negative_feedback", "anchor_click", "poi_detail", "poi_favorite",
    "conversion", "returned_next_session", "accepted_commercialization",
)
EVALUATION_VALUE_NAMES = (
    "lt_stay_component", "lt_active_day_component",
    "lt_accepted_commercialization_component", "lt_total_evaluation_only",
)
SEQUENCE_FIELDS = (
    "topic_norm", "stay_norm", "long_view", "quality_long_view", "like",
    "negative_feedback", "anchor_click", "conversion",
)


@dataclass(frozen=True)
class V3LoggingConfig:
    users: int = 50_000
    steps: int = 24
    batch_users: int = 25_000
    epsilon: float = 0.20
    sequence_length: int = 24
    seed: int = 20260823
    device: str = "cuda:0"

    def __post_init__(self):
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("logging epsilon must be inside (0, 1)")


def _split(step: int, steps: int) -> str:
    ratio = step / max(steps, 1)
    return "train" if ratio < 0.60 else ("validation" if ratio < 0.80 else "test")


def _choice(config, user_ids, step, scores, eligible):
    greedy = scores.argmax(dim=1)
    explore = uniform(user_ids, step, 80, config.seed) < config.epsilon
    eligible_count = eligible.sum(dim=1).clamp_min(1)
    random_rank = torch.floor(
        uniform(user_ids, step, 81, config.seed) * eligible_count
    ).long().clamp_max(eligible_count - 1)
    eligible_rank = eligible.long().cumsum(dim=1) - 1
    random_choice = (
        (eligible_rank == random_rank[:, None]) & eligible
    ).long().argmax(dim=1)
    choice = torch.where(explore, random_choice, greedy)
    base = config.epsilon / eligible_count
    propensity = torch.where(
        choice == greedy,
        1.0 - config.epsilon + base,
        base,
    )
    return choice, greedy, propensity


def _labels(values, selected, returned):
    conversion = values["paid"] | values["pixel"]
    completion = values["stay"] / selected["duration"].clamp_min(1.0)
    return torch.stack((
        values["played"].float(),
        (values["stay"] >= 3.0).float(),
        values["stay"],
        completion.clamp(0.0, 1.0),
        (completion >= 0.95).float(),
        values["long_view"].float(),
        values["quality_view"].float(),
        values["like"].float(),
        values["negative"].float(),
        values["anchor"].float(),
        values["detail"].float(),
        values["favorite"].float(),
        conversion.float(),
        returned.float(),
        values["accepted_commercialization"],
    ), dim=1)


def _evaluation_values(values, return_value):
    stay = values["stay"] / 60.0
    commercialization = values["accepted_commercialization"]
    return torch.stack((
        stay,
        return_value,
        commercialization,
        stay + return_value + commercialization,
    ), dim=1)


def _sequence_event(selected, values, config):
    conversion = values["paid"] | values["pixel"]
    return torch.stack((
        selected["candidate_topic"].float() / 11.0,
        torch.log1p(values["stay"]) / torch.log(
            torch.tensor(181.0, device=values["stay"].device)
        ),
        values["long_view"].float(),
        values["quality_view"].float(),
        values["like"].float(),
        values["negative"].float(),
        values["anchor"].float(),
        conversion.float(),
    ), dim=1)


def _append(storage, name, value, mask, dtype=None):
    selected = value[mask].detach().to("cpu")
    storage.setdefault(name, []).append(selected.to(dtype) if dtype else selected)


def _capture_step(storage, config, state, candidates, selected, features, labels,
                  evaluation_values, sequence, propensity, session_id, active, step):
    split = _split(step, config.steps)
    bucket = storage.setdefault(split, {})
    users = state["user_ids"]
    request_id = users * config.steps + step
    impression = 1_650_000_000_000 + step * 60_000 + torch.remainder(users, 60_000)
    audit_utility = torch.einsum(
        "bkd,bd->bk", candidates["topics"], state["interest"]
    ) + 0.45 * candidates["quality"]
    fields = {
        "request_id": (request_id, torch.int64),
        "user_id": (users, torch.int32),
        "request_step": (torch.full_like(users, step), torch.int16),
        "session_id": (session_id, torch.int16),
        "lifecycle_bucket": (state["lifecycle_bucket"], torch.uint8),
        "region_bucket": (state["region_bucket"], torch.uint8),
        "account_age_days": (state["account_age_days"], torch.float32),
        "historical_activity": (state["historical_activity"], torch.float32),
        "impression_time_ms": (impression, torch.int64),
        "recalled_item_ids": (candidates["recalled_item_ids"], torch.int32),
        "recalled_route_bits": (candidates["recalled_route_bits"], torch.uint8),
        "recall_scores": (candidates["recalled_scores"], torch.float16),
        "recalled_coarse_scores": (selected["coarse_scores"], torch.float16),
        "candidate_item_ids": (candidates["item_ids"], torch.int32),
        "candidate_route_bits": (candidates["route_bits"], torch.uint8),
        "candidate_features": (features, torch.float16),
        "candidate_coarse_scores": (selected["coarse_scores"], torch.float16),
        "candidate_coarse_mask": (selected["coarse_mask"], torch.uint8),
        "candidate_fine_scores": (selected["fine_scores"], torch.float16),
        "candidate_mix_scores": (selected["mix_scores"], torch.float16),
        "candidate_audit_utility": (audit_utility, torch.float16),
        "exposed_index": (selected["final_choice"], torch.int16),
        "exposure_propensity": (propensity, torch.float32),
        "behavior_sequence": (sequence, torch.float16),
        "labels": (labels, torch.float32),
        "label_masks": (torch.ones_like(labels), torch.uint8),
        "evaluation_values": (evaluation_values, torch.float32),
        "candidate_label_mask": (
            torch.nn.functional.one_hot(
                selected["final_choice"], candidates["item_ids"].shape[1]
            ), torch.uint8,
        ),
        "audit_oracle_item": (candidates["audit_oracle_item"], torch.int32),
        "stage_attribution": (selected["stage_attribution"], torch.uint8),
    }
    for name, (value, dtype) in fields.items():
        _append(bucket, name, value, active, dtype)


def _simulate_batch(storage, config, tensor_config, catalog, generator, device, offset):
    users = min(config.batch_users, config.users - offset)
    user_ids = torch.arange(offset, offset + users, device=device)
    state = new_user_state(
        tensor_config, PERSONALIZED, generator, device, user_ids
    )
    sequence = torch.zeros(
        users, config.sequence_length, len(SEQUENCE_FIELDS), device=device
    )
    for step in range(config.steps):
        refresh_search_state(tensor_config, state, step)
        candidates = candidate_batch(
            tensor_config, generator, device, state, catalog, step, PERSONALIZED
        )
        features = build_tensor_features(
            tensor_config, user_ids, state, candidates, step
        )
        scores, affinity = _fine_score(
            PERSONALIZED, state["eligible"], user_ids, state, candidates
        )
        coarse_scores, coarse_mask, coarse_keep = coarse_rank(
            PERSONALIZED, affinity, candidates, tensor_config.candidates
        )
        scores = scores.masked_fill(~coarse_mask, -1e9)
        choice, greedy, propensity = _choice(
            config, user_ids, step, scores, coarse_mask
        )
        selected = materialize_selected(
            PERSONALIZED, user_ids, state, candidates, choice, greedy, scores,
            scores, coarse_scores, coarse_mask, coarse_keep, device,
        )
        active = state["active"].clone()
        session_id = state["sessions"].clone()
        values = sample_step(
            tensor_config, PERSONALIZED, generator, device, state, selected, step
        )
        return_value, returned = advance_state(
            tensor_config, PERSONALIZED, generator, state, selected, values, step
        )
        _capture_step(
            storage, config, state, candidates, selected, features,
            _labels(values, selected, returned),
            _evaluation_values(values, return_value), sequence, propensity,
            session_id, active, step,
        )
        sequence = torch.roll(sequence, shifts=-1, dims=1)
        sequence[:, -1] = _sequence_event(selected, values, config)


def _save(storage, output_dir, config, authority_bundle_id, storage_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for split, values in storage.items():
        tensors = {name: torch.cat(parts) for name, parts in values.items()}
        path = output_dir / f"{split}.pt"
        torch.save({"tensors": tensors}, path)
        tables[split] = {
            "path": path.name,
            "requests": len(tensors["request_id"]),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema": "v3-request-candidate-log-v1",
        "storage_root": storage_root,
        "config": asdict(config),
        "authority_bundle_id": authority_bundle_id,
        "label_names": LABEL_NAMES,
        "evaluation_value_names": EVALUATION_VALUE_NAMES,
        "training_contract": {
            "primitive_labels_only": True,
            "lt_total_is_training_label": False,
            "returned_next_session_requires_delayed_maturity": True,
            "evaluation_values_are_ab_outcomes_not_ranker_targets": True,
        },
        "sequence_fields": SEQUENCE_FIELDS,
        "candidate_label_contract": "only exposed candidate has mask=1",
        "tables": tables,
    }


@torch.inference_mode()
def build_v3_logging_dataset(root: Path, output_dir: Path, config: V3LoggingConfig):
    authority_path = root / "artifacts/releases/simulated-feed-control.json"
    authority = json.loads(authority_path.read_text())
    authority_bundle_id = authority["active_bundle_id"]
    tensor_config = TensorFeedConfig(
        users=config.users, steps=config.steps, batch_users=config.batch_users,
        seed=config.seed, device=config.device, signal_version="kuairand-calibrated-v3",
    )
    device, generator, catalog = prepare_run(tensor_config, None, 0, None)
    storage = {}
    for offset in range(0, config.users, config.batch_users):
        _simulate_batch(
            storage, config, tensor_config, catalog, generator, device, offset
        )
    try:
        storage_root = str(output_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        storage_root = "external://v3-request-log"
    manifest = _save(storage, output_dir, config, authority_bundle_id, storage_root)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    attach_dataset(root, manifest)
    return manifest
