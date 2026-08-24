"""Generate a frozen request-level training snapshot from the V3 candidate graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch

from ...authority import attach_dataset
from ...world_model.contracts import WORLD_LABEL_NAMES
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
from ..tensor_runtime.ranking_sequence import SEQUENCE_FIELDS
from .validation import validate_request_tensors


LABEL_NAMES = WORLD_LABEL_NAMES
EVALUATION_VALUE_NAMES = (
    "lt_stay_component", "lt_active_day_component",
    "lt_accepted_commercialization_component", "lt_total_evaluation_only",
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
    signal_version: str = "kuairand-calibrated-v3"
    local_signal_version: str = "legacy-local-v1"
    candidates: int = 20
    route_candidates: int = 8
    route_oversample: int = 3
    merged_candidates: int = 48
    audit_candidates: int = 24
    catalog_items: int = 200_000
    catalog_creators: int = 25_000

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
    labels = torch.stack((
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
        torch.zeros_like(values["stay"]),
        values["comment"].float(),
        values["share"].float(),
        values["follow"].float(),
        values["played"].float(),
        values["long_view"].float(),
    ), dim=1)
    masks = torch.ones_like(labels)
    masks[:, 15] = 0.0
    masks[:, 14] = values["ad_selected"].float()
    return labels, masks


def _evaluation_values(values, return_value):
    stay = values["stay"] / 60.0
    commercialization = values["accepted_commercialization"]
    return torch.stack((
        stay,
        return_value,
        commercialization,
        stay + return_value + commercialization,
    ), dim=1)


def _append(storage, name, value, mask, dtype=None):
    selected = value[mask].detach().to("cpu")
    storage.setdefault(name, []).append(selected.to(dtype) if dtype else selected)


def _capture_step(
    storage, config, state, candidates, selected, features, labels, label_masks,
    evaluation_values, sequence, propensity, session_id, active, audit_utility,
    step,
):
    split = _split(step, config.steps)
    bucket = storage.setdefault(split, {})
    users = state["user_ids"]
    request_id = users * config.steps + step
    impression = 1_650_000_000_000 + step * 60_000 + torch.remainder(users, 60_000)
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
        "label_masks": (label_masks, torch.uint8),
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


def _simulate_batch(
    storage, config, tensor_config, catalog, generator, device, offset,
    behavior_world=None,
):
    users = min(config.batch_users, config.users - offset)
    user_ids = torch.arange(offset, offset + users, device=device)
    state = new_user_state(
        tensor_config, PERSONALIZED, generator, device, user_ids
    )
    if behavior_world is not None:
        behavior_world.initialize_state(state)
    else:
        state["ranking_behavior_sequence"] = torch.zeros(
            users, config.sequence_length, len(SEQUENCE_FIELDS), device=device
        )
    sequence = state["ranking_behavior_sequence"]
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
        sequence = sequence.clone()
        audit_utility = (
            behavior_world.score_candidates(state, candidates, step)["utility"]
            if behavior_world is not None else torch.einsum(
                "bkd,bd->bk", candidates["topics"], state["interest"]
            ) + 0.45 * candidates["quality"]
        )
        values = sample_step(
            tensor_config, PERSONALIZED, generator, device, state, selected, step,
            behavior_world,
        )
        return_value, returned = advance_state(
            tensor_config, PERSONALIZED, generator, state, selected, values, step
        )
        labels, label_masks = _labels(values, selected, returned)
        _capture_step(
            storage, config, state, candidates, selected, features,
            labels, label_masks,
            _evaluation_values(values, return_value), sequence, propensity,
            session_id, active, audit_utility, step,
        )
        sequence = state["ranking_behavior_sequence"]


def _save(
    storage, output_dir, config, authority_bundle_id, storage_root,
    behavior_world=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    validation = {}
    for split, values in storage.items():
        tensors = {name: torch.cat(parts) for name, parts in values.items()}
        validation[split] = validate_request_tensors(tensors, config)
        path = output_dir / f"{split}.pt"
        torch.save({"tensors": tensors}, path)
        tables[split] = {
            "path": path.name,
            "requests": len(tensors["request_id"]),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema": (
            "v3-request-candidate-log-v1" if behavior_world is None
            else "v4-request-candidate-log-v1"
        ),
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
        "behavior_world": (
            None if behavior_world is None else behavior_world.describe()
        ),
        "tables": tables,
        "validation": validation,
    }


@torch.inference_mode()
def build_v3_logging_dataset(root: Path, output_dir: Path, config: V3LoggingConfig):
    authority_path = root / "artifacts/releases/simulated-feed-control.json"
    authority = json.loads(authority_path.read_text())
    authority_bundle_id = authority["active_bundle_id"]
    tensor_config = TensorFeedConfig(
        users=config.users, steps=config.steps, batch_users=config.batch_users,
        seed=config.seed, device=config.device,
        signal_version=config.signal_version,
        local_signal_version=config.local_signal_version,
        candidates=config.candidates, route_candidates=config.route_candidates,
        route_oversample=config.route_oversample,
        merged_candidates=config.merged_candidates,
        audit_candidates=config.audit_candidates,
        catalog_items=config.catalog_items,
        catalog_creators=config.catalog_creators,
        behavior_sequence_length=config.sequence_length,
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


@torch.inference_mode()
def build_v4_logging_dataset(
    root: Path, output_dir: Path, config: V3LoggingConfig, behavior_world,
):
    release_path = root / "artifacts/releases/simulator-world.json"
    release_hash = sha256(release_path.read_bytes()).hexdigest()
    release = json.loads(release_path.read_text())
    active = release["active_components"]["feed_behavior"]
    if active["response_world_artifact_sha256"] != behavior_world.artifact_sha256:
        raise ValueError("V4 logging world differs from simulator authority")
    tensor_config = TensorFeedConfig(
        users=config.users, steps=config.steps, batch_users=config.batch_users,
        seed=config.seed, device=config.device,
        signal_version=config.signal_version,
        local_signal_version=config.local_signal_version,
        candidates=config.candidates, route_candidates=config.route_candidates,
        route_oversample=config.route_oversample,
        merged_candidates=config.merged_candidates,
        audit_candidates=config.audit_candidates,
        catalog_items=config.catalog_items,
        catalog_creators=config.catalog_creators,
        behavior_sequence_length=config.sequence_length,
    )
    device, generator, catalog = prepare_run(
        tensor_config, None, 0, None, behavior_world
    )
    storage = {}
    for offset in range(0, config.users, config.batch_users):
        _simulate_batch(
            storage, config, tensor_config, catalog, generator, device, offset,
            behavior_world,
        )
    try:
        storage_root = str(output_dir.resolve().relative_to(root.resolve()))
    except ValueError:
        storage_root = "external://v4-request-log"
    authority_id = f"sha256:{release_hash}"
    manifest = _save(
        storage, output_dir, config, authority_id, storage_root, behavior_world
    )
    manifest["simulator_world_release_sha256"] = release_hash
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    return manifest
